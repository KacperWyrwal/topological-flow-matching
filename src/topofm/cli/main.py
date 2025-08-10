import os
from torch_ema import ExponentialMovingAverage
from dataclasses import dataclass
from typing import Optional
from everything import SDESolver
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from topofm import *


def _build_model(cfg: DictConfig, data_dim: int, laplacian: Optional[torch.Tensor] = None) -> torch.nn.Module:
    if cfg.model.name == "residual_nn":
        return ResidualNN(data_dim=data_dim, hidden_dim=cfg.model.hidden_dim, time_embed_dim=cfg.model.time_embed_dim, num_res_block=cfg.model.num_res_block)
    elif cfg.model.name == "gcn":
        assert laplacian is not None, "laplacian is required for GCN"
        return GCN(laplacian=laplacian, hidden_dim=cfg.model.hidden_dim, time_embed_dim=cfg.model.time_embed_dim)
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def _build_sde(cfg: DictConfig, eigenvalues: torch.Tensor) -> HeatBMTSDE:
    return HeatBMTSDE(eigenvalues=eigenvalues, c=cfg.sde.c, sigma=torch.tensor(cfg.sde.sigma))


def _build_laplacian(cfg: DictConfig) -> torch.Tensor:
    if cfg.data.name == 'brain':
        return load_brain_laplacian()
    raise ValueError(f"Unsupported tensor dataset name: {cfg.data.name}")


def _build_frame(cfg: DictConfig) -> torch.Tensor:
    if cfg.sde.topological is False:
        return StandardFrame()
    else:
        L = _build_laplacian(cfg)
        return SpectralFrame(L)


def _build_dataset(cfg: DictConfig, frame: SpectralFrame | None = None):
    frame = _build_frame(cfg)
    if cfg.data.name == "brain":
        x0, x1 = load_brain_data() # TODO pass option to retrieve the training split
        mu0, mu1 = Empirical(x0), Empirical(x1)
        mu0, mu1 = InFrame(mu0, frame), InFrame(mu1, frame)
        return MatchingDataset(mu0, mu1)
    raise ValueError(f"Unknown dataset: {cfg.data.name}")


def _build_sde_solver(cfg: DictConfig, sde: SDE) -> SDESolver:
    if cfg.solver.name == 'euler_maruyama':
        return EulerMaruyamaSolver(sde=sde, time_steps=cfg.solver.time_steps)
    else:
        raise NotImplementedError


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer.name == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=cfg.optimizer.lr)


def _build_ema(cfg: DictConfig, model: torch.nn.Module) -> ExponentialMovingAverage:
    return ExponentialMovingAverage(parameters=model.parameters(), decay=cfg.ema.decay)


def _build_objective(cfg: DictConfig):
    return torch.nn.MSELoss()


def _build_ot_solver(cfg: DictConfig, sde: SDE) -> OTSolver:
    return OTSolver(sde=sde, normalize_variance=cfg.ot_solver.normalize_variance)


def _build_time_sampler(cfg: DictConfig) -> TimeSampler:
    if cfg.time_sampler.name == 'uniform':
        return UniformTimeSampler()
    if cfg.time_sampler.name == 'discrete':
        time_steps = UniformTimeSteps(n=cfg.time_sampler.num_steps, t0=cfg.time_sampler.t0)
        return DiscreteTimeSampler(time_steps=time_steps)
    raise NotImplementedError


def _build_coupling(cfg: DictConfig, dataset: MatchingDataset, sde: SDE) -> Coupling:
    mu0, mu1 = dataset.mu0, dataset.mu1
    if cfg.coupling.name == 'independent':
        return IndependentCoupling(mu0, mu1)
    if cfg.coupling.name == 'ot':
        ot_solver = _build_ot_solver(cfg, sde=sde)
        return OTCoupling(mu0, mu1, ot_solver=ot_solver)
    if cfg.coupling.name == 'online_ot':
        ot_solver = _build_ot_solver(cfg, sde=sde)
        return OnlineOTCoupling(mu0, mu1, ot_solver=ot_solver)


def _build_train_data_loader(cfg: DictConfig, dataset: MatchingDataset, sde: SDE) -> MatchingDataLoader:
    if cfg.data.task == 'matching':
        coupling = _build_coupling(cfg, dataset=dataset, sde=sde)
        return MatchingDataLoader(
            coupling=coupling, 
            batch_size=cfg.train.batch_size, 
            num_batches=cfg.train.num_batches, # TODO set this in dataset config
        )
    raise NotImplementedError


def _build_eval_data_loader(cfg: DictConfig, dataset: MatchingDataset) -> DataLoader:
    if cfg.data.task == 'matching':
        dataset = to_tensor_dataset(dataset) # TODO Implement either in data or in utils. 
        return DataLoader(
            dataset=dataset, 
            batch_size=cfg.train.batch_size,
            shuffle=False, # No need to shuffle validation data
        )
    if cfg.data.task == 'generation':
        raise NotImplementedError 
    raise ValueError


def _build_test_data_loader(cfg: DictConfig, dataset: MatchingDataset) -> DataLoader:
    if cfg.data.task == 'matching':
        dataset = to_tensor_dataset(dataset)
        return DataLoader(
            dataset=dataset, 
            batch_size=cfg.train.batch_size, 
            shuffle=False, # No need to shuffle test data 
        )
    if cfg.data.task == 'generation':
        raise NotImplementedError
    raise ValueError


def run_test(
    cfg: DictConfig, 
    frame: Frame, 
    train_dataset: MatchingDataset, 
    test_dataset: MatchingDataset, 
    sde: SDE, 
) -> torch.nn.Module:
    model = _build_model(cfg, ...)
    sde_solver = _build_sde_solver(cfg, sde=sde)
    optimizer = _build_optimizer(cfg, model=model)
    ema = _build_ema(cfg, model=model)
    objective = _build_objective(cfg)

    train_dataset, eval_dataset = train_dataset.train_test_split(cfg.train.eval_size)

    train_data_loader = _build_train_data_loader(cfg, dataset=train_dataset, sde=sde)
    eval_data_loader = _build_eval_data_loader(cfg, dataset=eval_dataset)
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

    test_metrics = ... 

    # Save test metrics as csv 

    # Save predictions as image 
    predictions = ... 

    return 


def run_validation(
    cfg: DictConfig,
    *, 
    dataset: MatchingDataset, 

) -> torch.nn.Module:

    model = 
    metric_name = cfg.mode.validation.metric
    model = _build_model(cfg, ...)
    early_stopping = cfg.trainer.early_stopping 
    wnb_logger = ...
  
    for ds in dataset.chunk(k):
        ds_train, ds_eval = ds.train_test_split(eval_size)
        history = fit(...)
        # evaluate the model 
    return 



@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    assert cfg.data.type == 'tensor', "Currently only tensor datasets are supported."
    assert cfg.data.task == 'matching', "Currently only matching tasks are support."

    # Prelimiaries 
    torch.manual_seed(cfg.run.seed)
    torch.set_default_device(torch.device(cfg.run.device))
    torch.set_default_dtype(torch.dtype(cfg.run.dtype))

    # Common terms of full and cross-validation runs 
    frame = _build_frame(cfg)
    dataset = _build_dataset(cfg, frame=frame)
    dataset_train, dataset_test = dataset.train_test_split(test_size)
    sde = _build_sde(cfg, frame.eigenvalues)

    if cfg.train.validation is False: # TODO Maybe instead have cfg.train.mode == 'validation' and then a separate config for that mode 
        run_validation(cfg, dataset=dataset, sde=sde)
    else:
        run_test(cfg, ...)

    # TODO 


if __name__ == "__main__":
    main()


