from torch_ema import ExponentialMovingAverage
from typing import Optional
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Fix circular imports by using specific imports
from ..data import (
    load_brain_data, 
    load_brain_laplacian,
    MatchingDataset,
    TimeSampler,
    UniformTimeSampler,
    DiscreteTimeSampler,
    MatchingTrainLoader,
    MatchingTestLoader,
    EmpiricalToEmpiricalTestLoader,
    AnalyticToAnalyticTestLoader,
)
from ..frames import (
    Frame,
    StandardFrame, 
    SpectralFrame,
)
from ..sde import (
    SDE,
    HeatBMTSDE,
)
from ..sde_solvers import (
    SDESolver,
    EulerMaruyamaSolver,
)
from ..models import (
    ResidualNN, 
    GCN,
)
from ..distributions import (
    Empirical, 
    EmpiricalInFrame, 
    AnalyticInFrame,
    EightGaussians, 
    Moons,
)
from ..coupling import (
    Coupling,
    IndependentCoupling,
    OTCoupling,
    OnlineOTCoupling,
)
from ..ot import (
    OTSolver,
)
from ..time import (
    UniformTimeSteps,
)
from ..training import (
    fit, 
    evaluate,
)


def _get_dtype(cfg: DictConfig) -> torch.dtype:
    dtype_map = {
        'float32': torch.float32, 
        'float64': torch.float64, 
    }
    return dtype_map[cfg.run.dtype]


def _build_laplacian(cfg: DictConfig) -> torch.Tensor:
    if cfg.data.name == 'brain':
        return load_brain_laplacian()
    raise ValueError(f"Unsupported tensor dataset name: {cfg.data.name}")


def _build_frame(cfg: DictConfig) -> torch.Tensor:
    if cfg.frame.name == 'standard':
        return StandardFrame()
    
    if cfg.frame.name == 'spectral':
        L = _build_laplacian(cfg)
        return SpectralFrame(L)


def _build_sde(cfg: DictConfig, eigenvalues: torch.Tensor) -> HeatBMTSDE:
    if cfg.sde.name == 'heat_bm':
        c = cfg.sde.c
        sigma = torch.as_tensor(cfg.sde.sigma)
        return HeatBMTSDE(eigenvalues=eigenvalues, c=c, sigma=sigma)


def _build_model(cfg: DictConfig, data_dim: int, laplacian: Optional[torch.Tensor] = None) -> torch.nn.Module:
    if cfg.model.name == "residual_nn":
        return ResidualNN(
            data_dim=data_dim, 
            hidden_dim=cfg.model.hidden_dim, 
            time_embed_dim=cfg.model.time_embed_dim, 
            num_res_block=cfg.model.num_res_block,
        )
    elif cfg.model.name == "gcn":
        assert laplacian is not None, "laplacian is required for GCN"
        return GCN(
            laplacian=laplacian, 
            hidden_dim=cfg.model.hidden_dim, 
            time_embed_dim=cfg.model.time_embed_dim,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def _build_dataset(cfg: DictConfig, frame: SpectralFrame | None = None):
    frame = _build_frame(cfg)

    if cfg.data.name == "brain":
        x0, x1 = load_brain_data()
        mu0, mu1 = Empirical(x0), Empirical(x1)
        mu0, mu1 = EmpiricalInFrame(mu0, frame), EmpiricalInFrame(mu1, frame)
        return MatchingDataset(mu0, mu1)

    if cfg.data.name == "gaussians_to_moons":
        mu0 = EightGaussians(radius=cfg.data.radius, noise_std=cfg.data.gaussians_noise)
        mu1 = Moons(noise_std=cfg.data.moons_noise)
        mu0, mu1 = AnalyticInFrame(mu0, frame), AnalyticInFrame(mu1, frame)
        return MatchingDataset(mu0, mu1)

    raise ValueError(f"Unknown dataset: {cfg.data.name}")


def _build_sde_solver(cfg: DictConfig, sde: SDE) -> SDESolver:
    if cfg.sde_solver.name == 'euler_maruyama':
        return EulerMaruyamaSolver(sde=sde, time_steps=cfg.sde_solver.time_steps)
    else:
        raise NotImplementedError


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.train.optimizer.name == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=cfg.train.optimizer.lr)


def _build_ema(cfg: DictConfig, model: torch.nn.Module) -> ExponentialMovingAverage:
    return ExponentialMovingAverage(parameters=model.parameters(), decay=cfg.ema.decay)


def _build_objective(cfg: DictConfig):
    if cfg.train.objective.name == 'mse':
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported objective {cfg.train.objective.name}.")


def _build_ot_solver(cfg: DictConfig, sde: SDE) -> OTSolver:
    return OTSolver(sde=sde, normalize_variance=cfg.ot_solver.normalize_variance)


def _build_time_sampler(cfg: DictConfig) -> TimeSampler:
    if cfg.train.time_sampler.name == 'uniform':
        return UniformTimeSampler()
    if cfg.train.time_sampler.name == 'discrete':
        time_steps = UniformTimeSteps(
            n=cfg.train.time_sampler.num_steps, 
            t0=cfg.train.time_sampler.t0,
        )
        return DiscreteTimeSampler(time_steps=time_steps)
    raise NotImplementedError


def _build_coupling(cfg: DictConfig, dataset: MatchingDataset, sde: SDE) -> Coupling:
    mu0, mu1 = dataset.mu0, dataset.mu1
    if cfg.train.coupling.name == 'independent':
        return IndependentCoupling(mu0, mu1)

    if cfg.train.coupling.name == 'ot':
        ot_solver = _build_ot_solver(cfg, sde=sde)
        return OTCoupling(mu0, mu1, ot_solver=ot_solver)

    if cfg.train.coupling.name == 'online_ot':
        ot_solver = _build_ot_solver(cfg, sde=sde)
        return OnlineOTCoupling(mu0, mu1, ot_solver=ot_solver)


def _build_train_data_loader(cfg: DictConfig, dataset: MatchingDataset, sde: SDE) -> MatchingTrainLoader:
    coupling = _build_coupling(cfg, dataset=dataset, sde=sde)
    return MatchingTrainLoader(
        coupling=coupling, 
        batch_size=cfg.train.batch_size, 
        epoch_size=cfg.train.epoch_size, # TODO implement in MatchingTrainLoader
    )


def _build_test_data_loader(cfg: DictConfig, dataset: MatchingDataset) -> MatchingTestLoader:
    if cfg.data.task == 'empirical_to_empirical':
        return EmpiricalToEmpiricalTestLoader(
            dataset=dataset, 
            batch_size=cfg.batch_size, 
        )
    if cfg.data.task == 'analytic_to_analytic':
        return AnalyticToAnalyticTestLoader(
            dataset=dataset, 
            batch_size=cfg.test.batch_size, 
            epoch_size=cfg.test.epoch_size,
        )
    if cfg.data.task == 'analytic_to_empirical':
        raise NotImplementedError
    raise ValueError


def _build_eval_data_loader(cfg: DictConfig, dataset: MatchingDataset) -> MatchingTestLoader:
    if cfg.data.task == 'empirical_to_empirical':
        return EmpiricalToEmpiricalTestLoader(
            dataset=dataset, 
            batch_size=cfg.batch_size, 
        )
    if cfg.data.task == 'analytic_to_analytic':
        return AnalyticToAnalyticTestLoader(
            dataset=dataset, 
            batch_size=cfg.validation.batch_size, 
            epoch_size=cfg.validation.epoch_size,
        )
    if cfg.data.task == 'analytic_to_empirical':
        raise NotImplementedError
    raise ValueError



def run_test(
    cfg: DictConfig, 
    frame: Frame, 
    train_dataset: MatchingDataset, 
    test_dataset: MatchingDataset, 
    sde: SDE, 
) -> None:
    model = _build_model(cfg, ...)
    sde_solver = _build_sde_solver(cfg, sde=sde)
    optimizer = _build_optimizer(cfg, model=model)
    ema = _build_ema(cfg, model=model)
    objective = _build_objective(cfg)

    train_dataset, eval_dataset = train_dataset.train_test_split(cfg.validation.ratio)

    train_data_loader = _build_train_data_loader(cfg, dataset=train_dataset, sde=sde)
    eval_data_loader = _build_test_data_loader(cfg, dataset=eval_dataset)
    test_data_loader = _build_test_data_loader(cfg, dataset=test_dataset)
    time_sampler = _build_time_sampler(cfg)

    history = fit(
        sde=sde, 
        model=model, 
        train_data_loader=train_data_loader,
        eval_data_loader=eval_data_loader,
        time_sampler=time_sampler,
        num_epochs=cfg.train.num_epochs,
        sde_solver=sde_solver, 
        optimizer=optimizer,
        ema=ema,
        objective=objective,
    )

    test_metrics = evaluate(
        sde_solver=sde_solver, 
        model=model, 
        data_loader=test_data_loader, 
        ema=ema,
    ) 

    # TODO Add Plotting and saving to WnB
    print(test_metrics)


def run_validation() -> None:
    pass 



@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    assert cfg.data.task == 'empirical_to_empirical', "Currently only empirical-to-empirical tasks are supported."

    # Prelimiaries 
    torch.manual_seed(cfg.run.seed)
    torch.set_default_device(torch.device(cfg.run.device))
    torch.set_default_dtype(_get_dtype(cfg))

    # Common terms of full and cross-validation runs 
    frame = _build_frame(cfg)
    dataset = _build_dataset(cfg, frame=frame)
    train_dataset, test_dataset = dataset.train_test_split(cfg.test.ratio)
    sde = _build_sde(cfg, frame.eigenvalues)

    if cfg.run.mode == 'test':
        run_test(cfg, frame=frame, train_dataset=train_dataset, test_dataset=test_dataset, sde=sde)
    else:
        raise NotImplementedError

    # TODO 


if __name__ == "__main__":
    main()


