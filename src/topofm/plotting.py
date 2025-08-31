import numpy as np
import torch
import pandas as pd 
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import scanpy as sc
import os 
from anndata import AnnData
import plotly.express as px 
import plotly.graph_objects as go 
from .utils import single_cell_to_times
from .data import load_brain_regions_centroids


def plot_trajectory(
    xt: torch.Tensor,
    t: torch.Tensor,
    *,
    alpha: float = 0.1,
    ax: plt.Axes | None = None,
    linewidth: float = 1.0,
    cmap_name: str = "coolwarm",
    add_colorbar: bool = True,
):
    assert xt.ndim == 3 and xt.size(-1) == 2, f"xt must be a 3D tensor with shape (N, T, 2), got {xt.shape}"
    xt_np = xt.detach().cpu().numpy()
    t_np = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    if t_np.ndim == 2:
        t_np = t_np[0]
    assert t_np.ndim == 1 and t_np.shape[0] == xt_np.shape[1]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    N, T, _ = xt_np.shape
    if T < 2:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return fig, ax
    seg_start = xt_np[:, :-1, :]
    seg_end = xt_np[:, 1:, :]
    all_segments = np.stack((seg_start, seg_end), axis=2).reshape(-1, 2, 2)
    t_mid = (t_np[:-1] + t_np[1:]) * 0.5
    norm = plt.Normalize(0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    base_colors = cmap(norm(t_mid))
    base_colors[:, -1] = alpha
    all_colors = np.tile(base_colors, (N, 1))
    lc = LineCollection(all_segments, colors=all_colors, linewidths=linewidth)
    ax.add_collection(lc)
    ax.set_xlim(np.min(xt_np[..., 0]), np.max(xt_np[..., 0]))
    ax.set_ylim(np.min(xt_np[..., 1]), np.max(xt_np[..., 1]))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Sample Paths")
    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Time")
    return fig, ax


def plot_samples(
    x: torch.Tensor,
    *,
    alpha: float = 0.5,
    ax: plt.Axes | None = None,
    t: torch.Tensor | float | None = None,
    cmap_name: str = "coolwarm",
    color: str = 'k',
    marker='o',
    label: str | None = None,
    s=20,
    add_colorbar: bool = False,
):
    x_np = x.cpu().numpy()
    if isinstance(t, torch.Tensor):
        t = t.cpu().numpy()
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    if t is not None:
        label = f"t={t:.3f}" if isinstance(t, float) else label
        norm = plt.Normalize(0.0, 1.0)
        cmap = plt.get_cmap(cmap_name)
        color = cmap(norm(t if isinstance(t, float) else float(t)))
    ax.scatter(x_np[:, 0], x_np[:, 1], marker=marker, s=s, color=color, alpha=alpha, label=label)
    if add_colorbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Time")
    return fig, ax


def plot_history(history: dict[str, list[float]]) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot training history using matplotlib.

    Args:
        history: Dictionary with keys like 'loss', 'W1', 'W2' and values as lists of floats.

    Returns:
        fig, ax: Matplotlib Figure and Axes objects.
    """
    import matplotlib.pyplot as plt

    epochs = range(len(next(iter(history.values())))) if history else []
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot loss on left y-axis
    if 'loss' in history:
        ax1.plot(epochs, history['loss'], label='Loss', color='blue', linestyle='-')
        ax1.set_ylabel('Loss', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
    else:
        ax1.set_ylabel('Loss')

    # Plot W1 and W2 on right y-axis if present
    ax2 = ax1.twinx()
    lines = []
    labels = []

    if 'W1' in history:
        l1, = ax2.plot(epochs, history['W1'], label='W1', color='red', linestyle=':')
        lines.append(l1)
        labels.append('W1')
    if 'W2' in history:
        l2, = ax2.plot(epochs, history['W2'], label='W2', color='green', linestyle='--')
        lines.append(l2)
        labels.append('W2')

    if 'W1' in history or 'W2' in history:
        ax2.set_ylabel('Validation Metrics (W1, W2)')
        ax2.tick_params(axis='y', labelcolor='black')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    ax1.set_xlabel('Epoch')
    ax1.set_title('Metrics vs Epoch')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, ax1


def plot_2d_predictions(
    t: torch.Tensor,
    xt: torch.Tensor,
    x0: torch.Tensor,
    x1: torch.Tensor
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot predicted sample paths and endpoints.

    Args:
        t: Time steps tensor.
        xt: Trajectory tensor of shape (N, T, 2).
        x0: Initial samples tensor.
        x1: Target samples tensor.

    Returns:
        Tuple of (figure, axes) from matplotlib.
    """
    fig, ax = plot_trajectory(xt=xt, t=t)
    plot_samples(x0, t=0.0, ax=ax)
    plot_samples(x1, t=1.0, ax=ax)
    plot_samples(xt[:, -1], color='green', label='predicted', ax=ax)
    return fig, ax


"""
Brain plotting
"""
def plot_brain_signal_3d(signal, width: int = 500, height: int = 500, marker_size: int = 5):
    df = load_brain_regions_centroids()
    if signal is not None:
        df = df.assign(signal=signal.detach().cpu().numpy())
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='signal')
    else:
        fig = px.scatter_3d(df, x='x', y='y', z='z')
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(width=width, height=height)
    fig.show()
    return fig 


def plot_brain_signal_2d(signal, width: int = 400, height: int = 400, marker_size: int = 6):
    df = load_brain_regions_centroids()
    if signal is not None:
        df = df.assign(signal=signal.detach().cpu().numpy())
        fig = px.scatter(df, x='x', y='y', color='signal')
    else:
        fig = px.scatter(df, x='x', y='y')
    fig.update_traces(marker_size=marker_size)
    fig.update_layout(width=width, height=height)
    fig.show()
    return fig 



"""
single cell plotting, maybe move to run 
"""

def as_ndarray(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else:
        return np.asarray(x)


def _plot_single_cell_data(
    sample_labels: pd.Categorical,
    X_phate: np.ndarray,
    *, 
    ax: plt.Axes | None = None,
    show_legend: bool = False,
    title: str | None = None,
) -> None:
    adata = AnnData(X=X_phate)
    adata.obsm["X_phate"] = X_phate
    adata.obs["sample_labels"] = sample_labels
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    else:
        fig = ax.get_figure()

    sc.pl.scatter(
        adata,
        basis="phate",
        color="sample_labels",
        ax=ax,
        show=False,
        title=title, 
        legend_loc="none" if not show_legend else "right margin",
    )
    
    return fig, ax


def plot_single_cell_predictions(
    x1_pred: torch.Tensor,
    *, 
    data_dir: str = './', 
) -> None:
    adata = sc.read_h5ad(os.path.join(data_dir, "ebdata_v3.h5ad"))
    X_phate = adata.obsm["X_phate"]
    sample_labels = adata.obs["sample_labels"].values
    
    # Create the predictions dataset
    times_pred = single_cell_to_times(
        x1_pred, 
        torch.as_tensor(adata.obs["sample_labels"].cat.codes.values, device='cpu'),
    )
    sample_labels_pred = pd.Categorical.from_codes(
        times_pred, 
        categories=adata.obs['sample_labels'].cat.categories,
    )

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    _plot_single_cell_data(
        sample_labels=sample_labels_pred,
        X_phate=X_phate,
        ax=axs[0],
        title="Predicted",
    )
    _plot_single_cell_data(
        sample_labels=sample_labels,
        X_phate=X_phate,
        ax=axs[1],
        show_legend=True,
        title="True",
    )
    return fig, axs
