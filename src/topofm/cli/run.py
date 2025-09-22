import os 
from torch_ema import ExponentialMovingAverage
import hydra
from omegaconf import DictConfig
import torch
from matplotlib import pyplot as plt
import wandb
from omegaconf import OmegaConf
from datetime import datetime

# Fix circular imports by using specific imports
from ..data import (
    AnalyticToAnalyticDataset,
    EmpiricalToEmpiricalDataset,
    load_brain_data, 
    load_brain_laplacian,
    load_earthquakes_data,
    load_earthquakes_laplacian,
    MatchingDataset,
    TimeSampler,
    UniformTimeSampler,
    DiscreteTimeSampler,
    MatchingTrainLoader,
    MatchingTestLoader,
    EmpiricalToEmpiricalTestLoader,
    AnalyticToAnalyticTestLoader,
    AnalyticToEmpiricalTestLoader,
    AnalyticToEmpiricalDataset,
    load_ocean_eigenpairs,
    load_traffic_data,
    load_traffic_laplacian,
    load_single_cell_true_times,
    load_single_cell_phate,
    load_single_cell_eigenpairs,
    load_single_cell_data,
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
    EdgeGP,
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
    EarlyStopping,
    fit__single_cell,
    evaluate__single_cell,
)

from ..control import (
    ModelControl, 
)

from ..plotting import (
    plot_2d_predictions,
    plot_history,
    plot_single_cell_predictions,
)

from ..wandb_logger import WandBLogger
from hydra.utils import to_absolute_path


def _get_dtype(cfg: DictConfig) -> torch.dtype:
    dtype_map = {
        'float32': torch.float32, 
        'float64': torch.float64, 
    }
    return dtype_map[cfg.run.dtype]


def _setup_wandb(cfg: DictConfig):
    """Setup W&B logging if enabled."""
    if not cfg.run.use_wandb:
        return None
    
    # Generate run name if not specified
    run_name = cfg.run.wandb.run_name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg.data.name}-{cfg.model.name}-{cfg.sde.name}-{cfg.train.coupling.name}-{cfg.run.mode}-{timestamp}"
    
    wandb.init(
        project=cfg.run.wandb.project,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.run.wandb.tags,
        entity=cfg.run.wandb.entity,
        job_type=cfg.run.wandb.job_type, 
        notes=cfg.run.wandb.notes,
        reinit='finish_previous',
    )

    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("val/*",   step_metric="epoch")

    return wandb.run


def _build_laplacian_eigenpairs(cfg: DictConfig) -> tuple[torch.Tensor, torch.Tensor]:
    data_dir = to_absolute_path(cfg.data.dir) if hasattr(cfg.data, 'dir') else None
    if cfg.data.name == 'ocean':
        return load_ocean_eigenpairs(data_dir=data_dir)
    if cfg.data.name == 'single_cell':
        return load_single_cell_eigenpairs(data_dir=data_dir)
    raise ValueError(f"Unsupported dataset: {cfg.data.name}")


def _build_laplacian(cfg: DictConfig) -> torch.Tensor:
    data_dir = to_absolute_path(cfg.data.dir) if hasattr(cfg.data, 'dir') else None
    if cfg.data.name == 'brain':
        L = load_brain_laplacian(data_dir=data_dir)
        return L.to(torch.get_default_dtype())
    if cfg.data.name == 'gaussians_to_moons':
        if cfg.data.laplacian == 'fully_connected':
            A = torch.tensor([
                [0.0, 1.0], 
                [1.0, 0.0]
            ])
            D = torch.diag_embed(torch.sum(A, dim=-1))
            L = D - A
            return L.to(torch.get_default_dtype())
        else:
            raise ValueError(f"Unsupported Laplacian type {cfg.data.laplacian} for dataset {cfg.data.name}")
    if cfg.data.name == 'earthquakes':
        L = load_earthquakes_laplacian(data_dir=data_dir)
        assert L.device == torch.get_default_device()
        return L.to(torch.get_default_dtype())
    if cfg.data.name in ['ocean', 'single_cell']:
        eigenvecs, eigenvalues = _build_laplacian_eigenpairs(cfg)
        return eigenvecs.to(torch.get_default_dtype()), eigenvalues.to(torch.get_default_dtype())

    if cfg.data.name == 'traffic':
        L = load_traffic_laplacian(data_dir=data_dir)
        assert L.device == torch.get_default_device()
        return L.to(torch.get_default_dtype())

    raise ValueError(f"Unsupported tensor dataset name: {cfg.data.name}")


def _build_frame(cfg: DictConfig) -> torch.Tensor:
    if cfg.frame.name == 'standard':
        frame = StandardFrame()
        print("✅ StandardFrame created")
        return frame

    if cfg.frame.name == 'spectral' and cfg.data.name == 'ocean':
        eigenpairs = _build_laplacian_eigenpairs(cfg)
        eigenvectors = torch.concat([eigenpairs['grad_vecs'], eigenpairs['curl_vecs'], eigenpairs['harm_vecs']], dim=0)
        eigenvalues = torch.concat([eigenpairs['grad_vals'], eigenpairs['curl_vals'], eigenpairs['harm_vals']], dim=0)
        eigenvectors = eigenvectors.mT
        print(f"✅ Eigenvalues loaded: shape={eigenvalues.shape}")
        print(f"✅ Eigenvectors loaded: shape={eigenvectors.shape}")
        eigenvectors = eigenvectors.to(torch.get_default_dtype())
        eigenvalues = eigenvalues.to(torch.get_default_dtype())
        frame = SpectralFrame(eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        print("✅ SpectralFrame created")
        return frame

    if cfg.frame.name == 'spectral' and cfg.data.name == 'single_cell':
        eigenvectors, eigenvalues = _build_laplacian(cfg)
        print(f"✅ Eigenvectors loaded: shape={eigenvectors.shape}")
        eigenvectors = eigenvectors.to(torch.get_default_dtype())
        eigenvalues = eigenvalues.to(torch.get_default_dtype())
        frame = SpectralFrame(eigenvalues=eigenvalues, eigenvectors=eigenvectors)
        print("✅ SpectralFrame created")
        return frame
    
    if cfg.frame.name == 'spectral' and cfg.data.name != 'ocean':    
        L = _build_laplacian(cfg)
        print(f"✅ Laplacian loaded: shape={L.shape}")
        L = L.to(torch.get_default_dtype())
        frame = SpectralFrame(L)
        print("✅ SpectralFrame created")
        return frame
    
    raise ValueError(f"Unsupported frame name: {cfg.frame.name}")


def _build_sde(cfg: DictConfig, eigenvalues: torch.Tensor) -> HeatBMTSDE:
    if cfg.sde.name == 'topological_heat_bm' or cfg.sde.name == 'euclidean_heat_bm':
        c = cfg.sde.c
        sigma = torch.as_tensor(cfg.sde.sigma)
        return HeatBMTSDE(eigenvalues=eigenvalues, c=c, sigma=sigma)


def _build_model(cfg: DictConfig, frame: Frame, data_dim: int) -> torch.nn.Module:
    if cfg.model.name == "residual_nn":
        model = ResidualNN(
            data_dim=data_dim, 
            hidden_dim=cfg.model.hidden_dim, 
            time_embed_dim=cfg.model.time_embed_dim, 
            num_res_block=cfg.model.num_res_block,
        )
        assert model.device == torch.get_default_device()
        return model
    elif cfg.model.name == "gnn":
        model = GCN(
            eigenvalues=frame.eigenvalues, 
            hidden_dim=cfg.model.hidden_dim, 
            time_embed_dim=cfg.model.time_embed_dim,
            n_layers=cfg.model.n_layers,
        ).to(torch.get_default_device())
        # assert model.device == torch.get_default_device()
        return model

        raise NotImplementedError("GCN support coming soon.")
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")


def _build_dataset(cfg: DictConfig, frame: Frame | None = None):
    print(f"🔍 Building dataset: name={cfg.data.name}, task={cfg.data.task}")
    data_dir = to_absolute_path(cfg.data.dir) if hasattr(cfg.data, 'dir') else None
    
    frame = _build_frame(cfg)
    print("✅ Frame built for dataset")

    if cfg.data.name == "brain":
        print("🧠 Loading brain dataset...")
        x0, x1 = load_brain_data(data_dir=data_dir)
        x0, x1 = x0.to(torch.get_default_dtype()), x1.to(torch.get_default_dtype())
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
        mu0 = EightGaussians(radius=cfg.data.radius, noise_std=cfg.data.gaussians_noise)
        print(f"✅ EightGaussians distribution created with radius={cfg.data.radius} and noise_std={cfg.data.gaussians_noise}")

        mu1 = Moons(noise_std=cfg.data.moons_noise)
        print(f"✅ Moons distribution created with noise_std={cfg.data.moons_noise}")        
        
        mu0, mu1 = AnalyticInFrame(mu0, frame), AnalyticInFrame(mu1, frame)
        print("✅ Wrapped EightGaussians and Moons in AnalyticInFrame")
        
        dataset = AnalyticToAnalyticDataset(mu0, mu1)
        print(f"✅ Dataset created: {type(dataset).__name__}")
        return dataset

    if cfg.data.name == 'earthquakes':
        print("📊 Creating Earthquakes dataset...")
        x1 = load_earthquakes_data(data_dir=data_dir)
        x1 = x1.to(torch.get_default_dtype())
        print(f"✅ Earthquake data loaded: x1 shape={x1.shape}")
        mu1 = Empirical(x1)
        print(f"✅ Empirical distribution created")
        mu1 = EmpiricalInFrame(mu1, frame)
        print(f"✅ Wrapped Empirical distribution in EmpiricalInFrame")

        mu0_mean = torch.zeros(x1.shape[-1:])
        mu0_std = torch.full_like(mu0_mean, 1.0)
        mu0 = torch.distributions.Normal(mu0_mean, mu0_std)
        mu0 = torch.distributions.Independent(mu0, 1)
        print("✅ Normal distribution created")
        # mu0 = AnalyticInFrame(mu0, frame)

        assert mu0.sample((1, )).device == mu1.sample((1, )).device == torch.get_default_device()

        dataset = AnalyticToEmpiricalDataset(mu0, mu1)
        print(f"✅ Dataset created: {type(dataset).__name__}")
        return dataset

    if cfg.data.name == 'traffic':
        print("📊 Creating Traffic dataset...")
        x1 = load_traffic_data(data_dir=data_dir)
        x1 = x1.to(torch.get_default_dtype())
        mu1 = Empirical(x1)
        mu1 = EmpiricalInFrame(mu1, frame)
        
        mu0_mean = torch.zeros(x1.shape[-1:])
        mu0_std = torch.full_like(mu0_mean, 1.0)
        mu0 = torch.distributions.Normal(mu0_mean, mu0_std)
        mu0 = torch.distributions.Independent(mu0, 1)
        print("✅ Normal distribution created")

        dataset = AnalyticToEmpiricalDataset(mu0, mu1)
        print(f"✅ Dataset created: {type(dataset).__name__}")
        return dataset

    if cfg.data.name == 'single_cell':
        print("📊 Creating Single-cell dataset...")
        x0, x1 = load_single_cell_data(data_dir=data_dir)
        x0, x1 = x0.to(torch.get_default_dtype()), x1.to(torch.get_default_dtype())
        print(f"✅ Single-cell data loaded: x0 shape={x0.shape}, x1 shape={x1.shape}")
        
        mu0 = Empirical(x0)
        mu0 = EmpiricalInFrame(mu0, frame)
        print(f"✅ Empirical distribution created")
        
        mu1 = Empirical(x1)
        mu1 = EmpiricalInFrame(mu1, frame)
        print(f"✅ Empirical distribution created")
        
        print("📦 Creating EmpiricalToEmpiricalDataset...")
        dataset = EmpiricalToEmpiricalDataset(mu0, mu1)
        print(f"✅ Dataset created: {type(dataset).__name__}")
        return dataset

    if cfg.data.name == 'ocean':
        print("📊 Creating Ocean dataset...")
        eigenpairs = load_ocean_eigenpairs(data_dir=data_dir)
        for key in eigenpairs:
            eigenpairs[key] = eigenpairs[key].to(torch.get_default_dtype())

        mu0 = EdgeGP(
            grad_vecs=eigenpairs['grad_vecs'],
            curl_vecs=eigenpairs['curl_vecs'],
            harm_vecs=eigenpairs['harm_vecs'],
            grad_vals=eigenpairs['grad_vals'],
            curl_vals=eigenpairs['curl_vals'],
            harm_vals=eigenpairs['harm_vals'],
            gp_type=cfg.data.initial_gp.type,
            harm_sigma=cfg.data.initial_gp.harm_sigma,
            grad_sigma=cfg.data.initial_gp.grad_sigma,
            curl_sigma=cfg.data.initial_gp.curl_sigma,
            harm_kappa=cfg.data.initial_gp.harm_kappa,
            grad_kappa=cfg.data.initial_gp.grad_kappa,
            curl_kappa=cfg.data.initial_gp.curl_kappa,
        )

        mu1 = EdgeGP(
            grad_vecs=eigenpairs['grad_vecs'],
            curl_vecs=eigenpairs['curl_vecs'],
            harm_vecs=eigenpairs['harm_vecs'],
            grad_vals=eigenpairs['grad_vals'],
            curl_vals=eigenpairs['curl_vals'],
            harm_vals=eigenpairs['harm_vals'],
            gp_type=cfg.data.terminal_gp.type,
            harm_sigma=cfg.data.terminal_gp.harm_sigma,
            grad_sigma=cfg.data.terminal_gp.grad_sigma,
            curl_sigma=cfg.data.terminal_gp.curl_sigma,
            harm_kappa=cfg.data.terminal_gp.harm_kappa,
            grad_kappa=cfg.data.terminal_gp.grad_kappa,
            curl_kappa=cfg.data.terminal_gp.curl_kappa,
        )
        
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


def _build_early_stopping(cfg: DictConfig) -> EarlyStopping | None:
    if not cfg.train.early_stopping.enabled:
        return None
    return EarlyStopping(
        patience=cfg.train.early_stopping.patience,
        metric_name=cfg.train.early_stopping.metric_name,
        min_delta=cfg.train.early_stopping.min_delta,
        mode=cfg.train.early_stopping.mode,
    )


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
        return AnalyticToEmpiricalTestLoader(
            dataset=dataset, 
            batch_size=cfg.test.batch_size, 
        )
    raise ValueError


def _build_eval_data_loader(cfg: DictConfig, dataset: MatchingDataset) -> MatchingTestLoader:
    if cfg.validation.enabled is False:
        return None
    
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
        return AnalyticToEmpiricalTestLoader(
            dataset=dataset, 
            batch_size=cfg.validation.batch_size, 
        )
    raise ValueError


@torch.inference_mode()
def _plot_predictions(cfg: DictConfig, model: torch.nn.Module, sde_solver: SDESolver, dataset: MatchingDataset, frame: Frame, wandb_run = None) -> None:
    if cfg.data.name == 'single_cell':
        data_dir = to_absolute_path(cfg.data.dir) if hasattr(cfg.data, 'dir') else None
        control = ModelControl(model)
        x0 = dataset.mu0.sample((1, ))
        x1_pred = sde_solver.pushforward(x0=x0, control=control)
        x1_pred = frame.inverse_transform(x1_pred)[0] # [D]
        fig, axs = plot_single_cell_predictions(x1_pred.cpu(), data_dir=data_dir)
    elif cfg.data.name == 'gaussians_to_moons':
        # Predict 
        control = ModelControl(model)
        x0, x1 = dataset.sample((cfg.plot.predictions.num_samples, )) # TODO add to config 
        xt, t = sde_solver.sample_path(x0=x0, control=control)
        x0, x1, xt = frame.inverse_transform(x0, x1, xt)

        # Plot
        fig, ax = plot_2d_predictions(t=t, xt=xt, x0=x0, x1=x1)
    else:
        raise ValueError(f"Dataset {cfg.data.name} not supported.")
    # Save 
    full_name = cfg.plot.predictions.name + ".png"
    full_path = os.path.join(cfg.plot.predictions.dir, full_name)
    os.makedirs(cfg.plot.predictions.dir, exist_ok=True)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Log to W&B if enabled
    if cfg.plot.predictions.log_to_wandb and wandb_run is not None:
        wandb_run.log({"predictions": wandb.Image(str(full_path))})
    return


def _plot_history(cfg: DictConfig, history: dict[str, list[float]], wandb_run = None) -> None:
    fig, ax = plot_history(history)
    full_name = cfg.plot.history.name + ".png"
    full_path = os.path.join(cfg.plot.history.dir, full_name)
    os.makedirs(cfg.plot.history.dir, exist_ok=True)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Log to W&B if enabled
    if cfg.plot.history.log_to_wandb and wandb_run is not None:
        wandb_run.log({"training_history": wandb.Image(str(full_path))})
    return 


def split_train_test(cfg: DictConfig, dataset: MatchingDataset) -> tuple[MatchingDataset, MatchingDataset]:
    if cfg.test.enabled is False:
        return dataset, None
    assert 1 >= cfg.test.ratio >= 0, "Test ratio must be between 0 and 1 if test is enabled"

    if cfg.data.name in ['earthquakes', 'single_cell', 'ocean']:
        return dataset, dataset
    
    if cfg.data.name in ['brain', 'traffic']:
        return dataset.train_test_split(cfg.test.ratio)

    if cfg.data.name == 'gaussians_to_moons':
        return dataset.train_test_split(cfg.test.ratio)
    
    raise ValueError(f"Unknown dataset: {cfg.data.name}")


def split_train_validation(cfg: DictConfig, dataset: MatchingDataset) -> tuple[MatchingDataset, MatchingDataset]:
    if cfg.validation.enabled is False:
        return dataset, None
    assert 1 >= cfg.validation.ratio >= 0, "Validation ratio must be between 0 and 1 if validation is enabled"

    if cfg.data.name in ['earthquakes', 'single_cell', 'ocean']:
        return dataset, dataset
    
    if cfg.data.name in ['brain', 'traffic']:
        return dataset.train_test_split(cfg.validation.ratio)

    if cfg.data.name == 'gaussians_to_moons':
        return dataset.train_test_split(cfg.validation.ratio)
    
    raise ValueError(f"Unknown dataset: {cfg.data.name}")


def _fit(
    cfg: DictConfig, 
    *,
    sde: SDE,
    model: torch.nn.Module,
    train_data_loader: MatchingTrainLoader,
    eval_data_loader: MatchingTestLoader,
    time_sampler: TimeSampler,
    sde_solver: SDESolver,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage,
    objective: torch.nn.Module,
    early_stopping: EarlyStopping,
    wandb_logger: WandBLogger,
    frame: Frame,
) -> None:
    if cfg.data.name == 'single_cell':
        data_dir = to_absolute_path(cfg.data.dir) if hasattr(cfg.data, 'dir') else None
        true_times = load_single_cell_true_times(data_dir=data_dir)
        phate = load_single_cell_phate(data_dir=data_dir)
        return fit__single_cell(
            sde=sde,
            model=model,
            true_times=true_times,
            phate=phate,
            train_data_loader=train_data_loader,
            eval_data_loader=eval_data_loader,
            time_sampler=time_sampler,
            num_epochs=cfg.train.num_epochs,
            sde_solver=sde_solver, 
            optimizer=optimizer,
            ema=ema,
            objective=objective,
            early_stopping=early_stopping,
            logger=wandb_logger,
            frame=frame,
        )
    else:
        return fit(
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
            early_stopping=early_stopping,
            logger=wandb_logger,
            frame=frame,
        )


def _evaluate(
    cfg: DictConfig,
    sde_solver: SDESolver,
    model: torch.nn.Module,
    data_loader: MatchingTestLoader,
    ema: ExponentialMovingAverage,
    frame: Frame,
) -> None:

    if cfg.data.name == 'single_cell':
        data_dir = to_absolute_path(cfg.data.dir) if hasattr(cfg.data, 'dir') else None
        true_times = load_single_cell_true_times(data_dir=data_dir)
        phate = load_single_cell_phate(data_dir=data_dir)
        return evaluate__single_cell(
            sde_solver=sde_solver,
            model=model,
            data_loader=data_loader,
            ema=ema,
            true_times=true_times,
            phate=phate,
            frame=frame,
        )
    else:
        return evaluate(
            sde_solver=sde_solver,
            model=model,
            data_loader=data_loader,
            ema=ema,
            frame=frame,
        )
    raise ValueError(f"Unknown dataset: {cfg.data.name}")


def run_test(
    cfg: DictConfig, 
    frame: Frame, 
    train_dataset: MatchingDataset, 
    test_dataset: MatchingDataset, 
    sde: SDE, 
) -> None:
    model = _build_model(cfg, frame=frame, data_dim=train_dataset.dim)
    print(f"✅ Model built: {type(model).__name__}")
    
    sde_solver = _build_sde_solver(cfg, sde=sde)
    print(f"✅ SDE solver built: {type(sde_solver).__name__}")
    
    optimizer = _build_optimizer(cfg, model=model)
    print(f"✅ Optimizer built: {type(optimizer).__name__}")
    
    ema = _build_ema(cfg, model=model)
    print(f"✅ EMA built: {type(ema).__name__}")
    
    objective = _build_objective(cfg)
    print(f"✅ Objective built: {type(objective).__name__}")

    train_dataset, eval_dataset = split_train_validation(cfg, train_dataset)
    print(f"✅ Train/eval split")


    if cfg.data.name in ('traffic', 'earthquakes'):
        # If we are in the generation from Gaussian case, 
        # set the mean and standard deviation of the Gaussian 
        # to be derived from the data
        train_mean = train_dataset.mu1.samples.mean(dim=0)
        train_dataset.mu0.base_dist.loc = train_mean
        eval_dataset.mu0.base_dist.loc = train_mean
        test_dataset.mu0.base_dist.loc = train_mean
        print("✅ Set mu0 mean to training mean")

        train_std = train_dataset.mu1.samples.std(dim=0)
        train_dataset.mu0.base_dist.scale = train_std
        eval_dataset.mu0.base_dist.scale = train_std
        test_dataset.mu0.base_dist.scale = train_std
        print("✅ Set mu0 std to training std")


    train_data_loader = _build_train_data_loader(cfg, dataset=train_dataset, sde=sde)
    print("✅ Train data loader built")
    
    eval_data_loader = _build_eval_data_loader(cfg, dataset=eval_dataset)
    print("✅ Eval data loader built")
    
    test_data_loader = _build_test_data_loader(cfg, dataset=test_dataset)
    print("✅ Test data loader built")
    
    time_sampler = _build_time_sampler(cfg)
    print(f"✅ Time sampler built: {type(time_sampler).__name__}")

    early_stopping = _build_early_stopping(cfg)
    if early_stopping is not None:
        print(f"✅ Early stopping enabled: patience={early_stopping.patience}, metric={early_stopping.metric_name}, min_delta={early_stopping.min_delta}")
    else:
        print("ℹ️ Early stopping disabled")

    wandb_run = _setup_wandb(cfg)
    wandb_logger = WandBLogger(wandb_run) if wandb_run is not None else None
    if wandb_logger and wandb_logger.is_enabled():
        print(f"✅ W&B logging enabled: {wandb_run.name}")
    else:
        print("ℹ️ W&B logging disabled")

    print(f"📊 Training config: epochs={cfg.train.num_epochs}, batch_size={cfg.train.batch_size}")
    history = _fit(
        cfg=cfg,
        sde=sde, 
        model=model, 
        train_data_loader=train_data_loader,
        eval_data_loader=eval_data_loader,
        time_sampler=time_sampler,
        sde_solver=sde_solver, 
        optimizer=optimizer,
        ema=ema,
        objective=objective,
        early_stopping=early_stopping,
        wandb_logger=wandb_logger,
        frame=frame,
    )
    print("✅ Training completed!")

    if cfg.plot.history.enabled:
        print("📊 Plotting training history...")
        _plot_history(cfg, history, wandb_run)
        print("✅ Training history plotted!")
    else:
        print("ℹ️ Not plotting training history (disabled in config).")

    print("🧪 Starting evaluation...")
    test_metrics = _evaluate(
        cfg=cfg,
        sde_solver=sde_solver, 
        model=model, 
        data_loader=test_data_loader, 
        ema=ema,
        frame=frame,
    ) 
    print("✅ Evaluation completed!")
    print("📊 Test metrics:")
    print(test_metrics)

    # Log training curves and test metrics via WandBLogger
    if wandb_logger and wandb_logger.is_enabled():
        wandb_logger.log_test(test_metrics)

    if cfg.plot.predictions.enabled:
        print("📊 Plotting predictions...")
        _plot_predictions(cfg, model=model, sde_solver=sde_solver, dataset=test_dataset, frame=frame, wandb_run=wandb_run)
        print("✅ Predictions plotted!")
    else:
        print("ℹ️ Not plotting predictions (disabled in config).")

    # Final W&B logging and cleanup
    if wandb_logger and wandb_logger.is_enabled():
        wandb_logger.finish()
        print("✅ W&B logging completed!")


def run_validation() -> None:
    pass 


@hydra.main(config_path="../../../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("🚀 Starting main function...")
    
    # Prelimiaries 
    torch.manual_seed(cfg.run.seed)
    torch.set_default_device(torch.device(cfg.run.device))
    torch.set_default_dtype(_get_dtype(cfg))
    print(f"✅ Preliminaries set: seed={cfg.run.seed}, device={torch.get_default_device()}, dtype={torch.get_default_dtype()}")

    # Common terms of full and cross-validation runs 
    frame = _build_frame(cfg)
    print(f"✅ Frame built: {type(frame).__name__}")
    
    dataset = _build_dataset(cfg, frame=frame)
    print(f"✅ Dataset built: {type(dataset).__name__}, dim={dataset.dim}")
    
    train_dataset, test_dataset = split_train_test(cfg, dataset)
    print(f"✅ Dataset split")
    
    sde = _build_sde(cfg, frame.eigenvalues)
    print(f"✅ SDE built: {type(sde).__name__}")

    print(f"🎯 Running mode: {cfg.run.mode}")
    if cfg.run.mode == 'test':
        run_test(cfg, frame=frame, train_dataset=train_dataset, test_dataset=test_dataset, sde=sde)
        print("✅ Test run completed!")
    else:
        print(f"❌ Unsupported mode: {cfg.run.mode}")
        raise NotImplementedError

    print("🎉 Main function completed successfully!")
    # TODO


if __name__ == "__main__":
    main()


