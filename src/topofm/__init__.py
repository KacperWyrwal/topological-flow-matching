"""Topological Flow Matching (topofm)

Modular package factoring the original everything.py into submodules.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("topofm")
except PackageNotFoundError:  # local editable or src layout without install
    __version__ = "0.0.0"

# Re-export common API for convenience
from .utils import (
    sample_moons,
    sample_eight_gaussians,
    matmul_many,
    torch_divmod,
    joint_multinomial,
    as_tensors,
)
from .distributions import (
    Moons, 
    EightGaussians, 
    PossiblyDegenerateNormal, 
    Empirical,
    InFrame,
    EmpiricalInFrame, 
    AnalyticInFrame,
    EdgeGP,
)
from .frames import (
    Frame, 
    SpectralFrame, 
    StandardFrame,
)
from .time import TimeSteps, UniformTimeSteps
from .sde import (
    SDE,
    DiagonalSDE,
    OUDiagonalSDE,
    PossiblyDegenerateOUDiagonalSDE,
    HeatBMTSDE,
)
from .control import (
    Control,
    ZeroControl,
    BridgeControl,
    ModelControl,
    bridge_control,
)
from .sde_solvers import SDESolver, EulerMaruyamaSolver
from .ot import OTSolver, wasserstein_distance
from .data import (
    TimeSampler,
    UniformTimeSampler,
    DiscreteTimeSampler,
    MatchingDataset,
    load_brain_data, 
    load_brain_laplacian,
    load_brain_regions_centroids,
    download_brain_regions_centroids,
    load_earthquakes_data,
    load_earthquakes_laplacian,
    load_ocean_eigenpairs,
)
from .models import (
    timestep_embedding,
    FCs,
    ResNet_FC,
    TimestepBlock,
    TimestepEmbedSequential,
    GCNLayer,
    GCNBlock,
    GCN,
    ResidualNN,
)
from .training import (
    EMA_DECAY,
    ADAMW_LR,
    WASSERSTEIN_DISTANCE_BLUR,
    EULER_MARUYAMA_NUM_STEPS,
    DISCRETE_TIME_SAMPLER_NUM_STEPS,
    OT_SOLVER_NORMALIZE_VARIANCE,
    make_optimizer,
    make_ema,
    make_objective,
    make_sde_solver,
    make_ot_solver,
    make_time_sampler,
    train,
    evaluate,
    fit,
)
from .plotting import (
    plot_trajectory, 
    plot_samples, 
    plot_2d_predictions,
    plot_history,
)

from .coupling import (
    Coupling,
    IndependentCoupling, 
    OTCoupling, 
    OnlineOTCoupling,
)


__all__ = [
    "__version__",
    # utils
    "sample_moons",
    "sample_eight_gaussians",
    "matmul_many",
    "torch_divmod",
    "joint_multinomial",
    "as_tensors",
    # distributions
    "Moons",
    "EightGaussians",
    "PossiblyDegenerateNormal",
    "Empirical",
    "InFrame",
    "EmpiricalInFrame", 
    "AnalyticInFrame",
    "EdgeGP",
    # frames
    "Frame",
    "SpectralFrame",
    "StandardFrame",
    # time
    "TimeSteps",
    "UniformTimeSteps",
    # sde
    "SDE",
    "DiagonalSDE",
    "OUDiagonalSDE",
    "PossiblyDegenerateOUDiagonalSDE",
    "HeatBMTSDE",
    # control
    "Control",
    "ZeroControl",
    "BridgeControl",
    "ModelControl",
    "bridge_control",
    # sde_solvers
    "SDESolver",
    "EulerMaruyamaSolver",
    # ot
    "OTSolver",
    "wasserstein_distance",
    # data
    "TimeSampler",
    "UniformTimeSampler",
    "DiscreteTimeSampler",
    "MatchingDataset",
    "load_brain_data",
    "load_brain_laplacian",
    "load_brain_regions_centroids",
    "download_brain_regions_centroids",
    "load_earthquakes_data",
    "load_earthquakes_laplacian",
    "load_ocean_eigenpairs",
    # models
    "timestep_embedding",
    "FCs",
    "ResNet_FC",
    "TimestepBlock",
    "TimestepEmbedSequential",
    "GCNLayer",
    "GCNBlock",
    "GCN",
    "ResidualNN",
    # training
    "EMA_DECAY",
    "ADAMW_LR",
    "WASSERSTEIN_DISTANCE_BLUR",
    "EULER_MARUYAMA_NUM_STEPS",
    "DISCRETE_TIME_SAMPLER_NUM_STEPS",
    "OT_SOLVER_NORMALIZE_VARIANCE",
    "make_optimizer",
    "make_ema",
    "make_objective",
    "make_sde_solver",
    "make_ot_solver",
    "make_time_sampler",
    "train",
    "evaluate",
    "fit",
    # plotting
    "plot_trajectory",
    "plot_samples",
    "plot_history",
    "plot_2d_predictions",
    # coupling
    "Coupling",
    "IndependentCoupling",
    "OTCoupling",
    "OnlineOTCoupling",
]


