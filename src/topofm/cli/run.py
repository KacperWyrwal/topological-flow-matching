from torch_ema import ExponentialMovingAverage
from typing import Optional
import hydra
from omegaconf import DictConfig
import torch

# Fix circular imports by using specific imports
from ..data import (
    AnalyticToAnalyticDataset,
    EmpiricalToEmpiricalDataset,
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
    print(f"🔧 Building frame: name={cfg.frame.name}")
    
    if cfg.frame.name == 'standard':
        print("📐 Creating StandardFrame...")
        frame = StandardFrame()
        print("✅ StandardFrame created")
        return frame
    
    if cfg.frame.name == 'spectral':
        print("🌊 Creating SpectralFrame...")
        print("🔍 Loading laplacian...")
        L = _build_laplacian(cfg)
        print(f"✅ Laplacian loaded: shape={L.shape}")
        frame = SpectralFrame(L)
        print("✅ SpectralFrame created")
        return frame
    
    raise ValueError(f"Unsupported frame name: {cfg.frame.name}")


def _build_sde(cfg: DictConfig, eigenvalues: torch.Tensor) -> HeatBMTSDE:
    if cfg.sde.name == 'heat_bm':
        c = cfg.sde.c
        sigma = torch.as_tensor(cfg.sde.sigma)
        return HeatBMTSDE(eigenvalues=eigenvalues, c=c, sigma=sigma)


def _build_model(cfg: DictConfig, data_dim: int) -> torch.nn.Module:
    if cfg.model.name == "residual_nn":
        return ResidualNN(
            data_dim=data_dim, 
            hidden_dim=cfg.model.hidden_dim, 
            time_embed_dim=cfg.model.time_embed_dim, 
            num_res_block=cfg.model.num_res_block,
        )
    elif cfg.model.name == "gcn":
        raise NotImplementedError("GCN support coming soon.")
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def _build_dataset(cfg: DictConfig, frame: SpectralFrame | None = None):
    print(f"🔍 Building dataset: name={cfg.data.name}, task={cfg.data.task}")
    
    frame = _build_frame(cfg)
    print("✅ Frame built for dataset")

    if cfg.data.name == "brain":
        print("🧠 Loading brain dataset...")
        x0, x1 = load_brain_data()
        print(f"✅ Brain data loaded: x0 shape={x0.shape}, x1 shape={x1.shape}")
        
        print("📊 Creating Empirical distributions...")
        mu0, mu1 = Empirical(x0), Empirical(x1)
        print("✅ Empirical distributions created")
        
        print("🔧 Wrapping in EmpiricalInFrame...")
        mu0, mu1 = EmpiricalInFrame(mu0, frame), EmpiricalInFrame(mu1, frame)
        print("✅ Frame wrapping completed")
        
        print("📦 Creating EmpiricalToEmpiricalDataset...")
        dataset = EmpiricalToEmpiricalDataset(mu0, mu1)
        print(f"✅ Dataset created: {type(dataset).__name__}")
        return dataset

    if cfg.data.name == "gaussians_to_moons":
        print("📊 Creating Gaussians to Moons dataset...")
        print(f"🔧 Creating EightGaussians with radius={cfg.data.radius}, noise_std={cfg.data.gaussians_noise}")
        mu0 = EightGaussians(radius=cfg.data.radius, noise_std=cfg.data.gaussians_noise)
        print("✅ EightGaussians distribution created")
        
        print(f"🔧 Creating Moons with noise_std={cfg.data.moons_noise}")
        mu1 = Moons(noise_std=cfg.data.moons_noise)
        print("✅ Moons distribution created")
        
        print("🔧 Wrapping in AnalyticInFrame...")
        mu0, mu1 = AnalyticInFrame(mu0, frame), AnalyticInFrame(mu1, frame)
        print("✅ Frame wrapping completed")
        
        print("📦 Creating AnalyticToAnalyticDataset...")
        dataset = AnalyticToAnalyticDataset(mu0, mu1)
        print(f"✅ Dataset created: {type(dataset).__name__}")
        return dataset

    print(f"❌ Unknown dataset: {cfg.data.name}")
    raise ValueError(f"Unknown dataset: {cfg.data.name}")


def _build_sde_solver(cfg: DictConfig, sde: SDE) -> SDESolver:
    if cfg.sde_solver.name == 'euler_maruyama':
        if cfg.sde_solver.time_steps.name == 'uniform':
            time_steps = UniformTimeSteps(
                n=cfg.sde_solver.time_steps.num_steps,
                t0=cfg.sde_solver.time_steps.t0,
            )
        else:
            raise ValueError("Invalid TimeSteps name.")
        return EulerMaruyamaSolver(sde=sde, time_steps=time_steps)
    else:
        raise NotImplementedError


def _build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.train.optimizer.name == 'adamw':
        return torch.optim.AdamW(params=model.parameters(), lr=cfg.train.optimizer.lr)


def _build_ema(cfg: DictConfig, model: torch.nn.Module) -> ExponentialMovingAverage:
    return ExponentialMovingAverage(parameters=model.parameters(), decay=cfg.train.ema.decay)


def _build_objective(cfg: DictConfig):
    if cfg.train.objective.name == 'mse':
        return torch.nn.MSELoss()
    raise ValueError(f"Unsupported objective {cfg.train.objective.name}.")


def _build_ot_solver(cfg: DictConfig, sde: SDE) -> OTSolver:
    return OTSolver(sde=sde, normalize_variance=cfg.ot.normalize_variance)


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
    print(f"🔗 Building coupling: name={cfg.train.coupling.name}")
    
    mu0, mu1 = dataset.mu0, dataset.mu1
    print(f"📊 Using distributions: mu0={type(mu0).__name__}, mu1={type(mu1).__name__}")
    
    if cfg.train.coupling.name == 'independent':
        print("🔧 Creating IndependentCoupling...")
        coupling = IndependentCoupling(mu0, mu1)
        print("✅ IndependentCoupling created")
        return coupling

    if cfg.train.coupling.name == 'ot':
        print("🔧 Creating OTCoupling...")
        print("⚙️ Building OT solver...")
        ot_solver = _build_ot_solver(cfg, sde=sde)
        print("✅ OT solver built")
        coupling = OTCoupling(mu0, mu1, ot_solver=ot_solver)
        print("✅ OTCoupling created")
        return coupling

    if cfg.train.coupling.name == 'online_ot':
        print("🔧 Creating OnlineOTCoupling...")
        print("⚙️ Building OT solver...")
        ot_solver = _build_ot_solver(cfg, sde=sde)
        print("✅ OT solver built")
        coupling = OnlineOTCoupling(mu0, mu1, ot_solver=ot_solver)
        print("✅ OnlineOTCoupling created")
        return coupling
    
    print(f"❌ Unknown coupling: {cfg.train.coupling.name}")
    raise ValueError(f"Unknown coupling: {cfg.train.coupling.name}")


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
            batch_size=cfg.test.batch_size, 
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
            batch_size=cfg.validation.batch_size, 
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
    print("🔨 Building model...")
    model = _build_model(cfg, data_dim=train_dataset.dim)
    print(f"✅ Model built: {type(model).__name__}")
    
    print("⚙️ Building SDE solver...")
    sde_solver = _build_sde_solver(cfg, sde=sde)
    print(f"✅ SDE solver built: {type(sde_solver).__name__}")
    
    print("🎯 Building optimizer...")
    optimizer = _build_optimizer(cfg, model=model)
    print(f"✅ Optimizer built: {type(optimizer).__name__}")
    
    print("📈 Building EMA...")
    ema = _build_ema(cfg, model=model)
    print(f"✅ EMA built: {type(ema).__name__}")
    
    print("🎯 Building objective...")
    objective = _build_objective(cfg)
    print(f"✅ Objective built: {type(objective).__name__}")

    print("✂️ Splitting train/eval datasets...")
    train_dataset, eval_dataset = train_dataset.train_test_split(cfg.validation.ratio)
    print(f"✅ Train/eval split")

    print("📚 Building data loaders...")
    train_data_loader = _build_train_data_loader(cfg, dataset=train_dataset, sde=sde)
    print("✅ Train data loader built")
    
    eval_data_loader = _build_eval_data_loader(cfg, dataset=eval_dataset)
    print("✅ Eval data loader built")
    
    test_data_loader = _build_test_data_loader(cfg, dataset=test_dataset)
    print("✅ Test data loader built")
    
    print("⏰ Building time sampler...")
    time_sampler = _build_time_sampler(cfg)
    print(f"✅ Time sampler built: {type(time_sampler).__name__}")

    print("🚀 Starting training...")
    print(f"📊 Training config: epochs={cfg.train.num_epochs}, batch_size={cfg.train.batch_size}")
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
    print("✅ Training completed!")

    print("🧪 Starting evaluation...")
    test_metrics = evaluate(
        sde_solver=sde_solver, 
        model=model, 
        data_loader=test_data_loader, 
        ema=ema,
    ) 
    print("✅ Evaluation completed!")

    # TODO Add Plotting and saving to WnB
    print("📊 Test metrics:")
    print(test_metrics)


def run_validation() -> None:
    pass 



@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("🚀 Starting main function...")
    
    # Prelimiaries 
    print("📊 Setting up preliminaries...")
    torch.manual_seed(cfg.run.seed)
    torch.set_default_device(torch.device(cfg.run.device))
    torch.set_default_dtype(_get_dtype(cfg))
    print(f"✅ Preliminaries set: seed={cfg.run.seed}, device={cfg.run.device}, dtype={cfg.run.dtype}")

    # Common terms of full and cross-validation runs 
    print("🔧 Building frame...")
    frame = _build_frame(cfg)
    print(f"✅ Frame built: {type(frame).__name__}")
    
    print("📦 Building dataset...")
    dataset = _build_dataset(cfg, frame=frame)
    print(f"✅ Dataset built: {type(dataset).__name__}, dim={dataset.dim}")
    
    print("✂️ Splitting dataset...")
    train_dataset, test_dataset = dataset.train_test_split(cfg.test.ratio)
    print(f"✅ Dataset split")
    
    print("🌊 Building SDE...")
    sde = _build_sde(cfg, frame.eigenvalues)
    print(f"✅ SDE built: {type(sde).__name__}")

    print(f"🎯 Running mode: {cfg.run.mode}")
    if cfg.run.mode == 'test':
        print("🧪 Starting test run...")
        run_test(cfg, frame=frame, train_dataset=train_dataset, test_dataset=test_dataset, sde=sde)
        print("✅ Test run completed!")
    else:
        print(f"❌ Unsupported mode: {cfg.run.mode}")
        raise NotImplementedError

    print("🎉 Main function completed successfully!")
    # TODO


if __name__ == "__main__":
    main()


