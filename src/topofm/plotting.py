import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import plotly.graph_objects as go


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
    assert xt.ndim == 3 and xt.size(-1) == 2
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


def plot_history(history: dict[str, list[float]]):
    import pandas as pd
    fig = go.Figure()
    df = pd.DataFrame(history)
    if 'loss' in df.columns:
        fig.add_trace(go.Scatter(y=df['loss'], x=df.index, mode='lines', name='Loss', line=dict(color='blue', dash='solid'), yaxis='y1'))
    if 'W1' in df.columns:
        fig.add_trace(go.Scatter(y=df['W1'], x=df.index, mode='lines', name='W1', line=dict(color='red', dash='dot'), yaxis='y2'))
    if 'W2' in df.columns:
        fig.add_trace(go.Scatter(y=df['W2'], x=df.index, mode='lines', name='W2', line=dict(color='green', dash='dash'), yaxis='y2'))
    fig.update_layout(
        title="Metrics vs Epoch",
        xaxis_title="Epoch",
        yaxis=dict(title="Loss", side="left"),
        yaxis2=dict(title="Validation Metrics (W1, W2)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        template="plotly_white",
        width=1000,
        height=500,
    )
    return fig


