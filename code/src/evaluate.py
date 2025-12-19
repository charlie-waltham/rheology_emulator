import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as colors
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import yaml
import torch
from torchmetrics.functional import mean_absolute_percentage_error


TRUE_COL = "True Values"
PRED_COL = "Predictions"


def load_df(csv_path: str, true_col: str = TRUE_COL, pred_col: str = PRED_COL) -> pd.DataFrame:
    """Load a CSV and ensure it contains the expected columns."""
    df = pd.read_csv(csv_path)
    missing = [col for col in (true_col, pred_col) if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def plot_qq(df: pd.DataFrame,
            true_col: str = TRUE_COL,
            pred_col: str = PRED_COL,
            quantiles: int = 200,
            xylim: Optional[float] = None,
            ax: Optional[plt.Axes] = None):
    """Create a QQ plot of predictions vs true values and return fig, ax."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    q = np.linspace(0.0, 1.0, quantiles)
    y_true = df[true_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    q_true = np.quantile(y_true, q)
    q_pred = np.quantile(y_pred, q)

    # compute MAE and bias for annotation/legend
    try:
        mae = float(np.mean(np.abs(y_pred - y_true)))
        bias = float(np.mean(y_pred - y_true))
    except Exception:
        mae = float('nan')
        bias = float('nan')

    ax.plot(q_true, q_pred, ".", alpha=0.6, label=f"QQ (MAE={mae:.1e}, bias={bias:.1e})")

    # Determine sensible axis limits: contain central 99.5% of combined data
    combined = np.concatenate([y_true.ravel(), y_pred.ravel()])
    if combined.size == 0:
        mn = 0.0
        mx = 1.0
    else:
        p_low, p_high = np.percentile(combined, [0.25, 99.75])
        mn = float(p_low)
        mx = float(p_high)
        if mn == mx:
            # expand a little if constant data
            mx = mn + 1e-6

    # draw 1:1 line across the plotted limits
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="1:1")

    # apply either user-supplied symmetric limit or percentile-based limits
    if xylim is not None:
        ax.set_xlim(-abs(xylim), abs(xylim))
        ax.set_ylim(-abs(xylim), abs(xylim))
    else:
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)

    fig.set_size_inches(10, 10)
    ax.set_xlabel("True quantiles")
    ax.set_ylabel("Pred quantiles")
    ax.set_title("QQ plot")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_hexbin(df: pd.DataFrame,
                true_col: str = TRUE_COL,
                pred_col: str = PRED_COL,
                gridsize: int = 60,
                extent: Optional[Tuple[float, float, float, float]] = None,
                ax: Optional[plt.Axes] = None):
    """Create a hexbin scatter plot with log color scale."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    hb = ax.hexbin(df[true_col], df[pred_col], gridsize=gridsize, bins='log', cmap='viridis', extent=extent)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(N)')

    mn = df[[true_col, pred_col]].min().min()
    mx = df[[true_col, pred_col]].max().max()
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1, label='1:1')

    fig.set_size_inches(10, 10)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predictions")
    ax.set_title("Hexbin (Pred vs True)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_hist(df: pd.DataFrame,
              true_col: str = TRUE_COL,
              pred_col: str = PRED_COL,
              bins: int = 80,
              x_range: Optional[Tuple[float, float]] = None,
              density: bool = True,
              ax: Optional[plt.Axes] = None):
    """Plot normalized histograms of true and predicted values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    # determine plotting range: if user provided x_range, use it; otherwise
    # compute central 99.5% coverage (percentiles 0.25 and 99.75) of combined data
    y_true = df[true_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    # combine true and pred into a single 1-D array; concatenating empty arrays yields empty array
    combined = np.concatenate([y_true.ravel(), y_pred.ravel()])

    if x_range is None:
        if combined.size == 0:
            xlo, xhi = -0.1, 0.1
        else:
            xlo, xhi = np.percentile(combined, [0.25, 99.75])
            if xlo == xhi:
                # ensure a non-zero width
                xlo -= 1e-6
                xhi += 1e-6
        x_range_use = (float(xlo), float(xhi))
    else:
        x_range_use = x_range

    ax.hist(y_true, bins=bins, range=x_range_use, density=density, alpha=0.6, label='True', color='C0')
    ax.hist(y_pred, bins=bins, range=x_range_use, density=density, alpha=0.6, label='Pred', color='C1')

    fig.set_size_inches(10, 10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title("Histogram (True vs Pred)")
    if x_range_use is not None:
        ax.set_xlim(x_range_use)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return fig, ax

def plot_polar_map(df: pd.DataFrame,
                   ds: xr.Dataset,
                   true_col: str = TRUE_COL,
                   pred_col: str = PRED_COL,
                   stride_base=800_000,
                   hemisphere="north",
                   resolution=1000,
                   lat_cutoff=80,
                   dist_threshold=10000):
        
        y_pred = torch.tensor(df[pred_col])
        y_true = torch.tensor(df[true_col])
        mape = torch.abs(y_pred - y_true) / torch.clamp(torch.abs(y_true), min=1.17e-06)

        stride = max(1, y_true.shape[0] // stride_base)
        mape = mape.numpy()[::stride]
        lat = ds.coords['lat'].values[::stride]
        lon = ds.coords['lon'].values[::stride]

        if hemisphere == "south":
            projection = ccrs.SouthPolarStereo()
            extent = [-180, 180, -lat_cutoff, -90]
        else:
            projection = ccrs.NorthPolarStereo()
            extent = [-180, 180, lat_cutoff, 90]
        
        # Source coord system is lon/lat
        src_crs = ccrs.PlateCarree()

        # Transform to meters
        coords_proj = projection.transform_points(src_crs, lon, lat)
        x_points = coords_proj[:, 0]
        y_points = coords_proj[:, 1]

        grid_x_2d, grid_y_2d = np.meshgrid(
            np.linspace(-4000000, 4000000, resolution),
            np.linspace(-4000000, 4000000, resolution)
        )

        grid_interpolated = griddata(
            (x_points, y_points),
            mape,
            (grid_x_2d, grid_y_2d),
            method='linear'
        )

        # Mask points that are dist_threshold from any datapoint
        tree = cKDTree(np.column_stack((x_points, y_points)))
        grid_pixels = np.column_stack((grid_x_2d.ravel(), grid_y_2d.ravel()))
        dist, _ = tree.query(grid_pixels)
        dist = dist.reshape(grid_x_2d.shape)
        grid_interpolated[dist > dist_threshold] = np.nan

        # Create plot and add continental features
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=projection)
        ax.set_extent(extent, src_crs)
        ax.add_feature(cfeature.LAND, zorder=2, facecolor='gray')
        ax.gridlines()

        # Create circular boundary
        theta = np.linspace(0, 2*np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        mesh = ax.pcolormesh(
            grid_x_2d, grid_y_2d, 
            grid_interpolated,
            transform=projection,
            norm=colors.LogNorm(),
            cmap='viridis',
            shading='auto'
        )

        plt.colorbar(mesh, ax=ax, label='Mean Absolute Percentage Error (%)', format=ticker.PercentFormatter(1))
        ax.set
        plt.title("Sea Ice Velocity MAPE Map")

        return fig, ax

def evaluate_and_save(csv_path: str, results_dir: str):
    """Load data, plot QQ/hexbin/hist, and save figures to results_dir."""
    df = load_df(csv_path)

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(results_dir, "used_training_config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    ds = xr.open_zarr(config["pairs_path"])
    indices = df["Dataset Indices"].to_numpy()
    ds = ds.isel(z=indices)

    # QQ
    fig, _ = plot_qq(df)
    qq_path = out_dir / "qq.png"
    fig.savefig(qq_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Hexbin
    fig, _ = plot_hexbin(df)
    hex_path = out_dir / "hexbin.png"
    fig.savefig(hex_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Histogram
    fig, _ = plot_hist(df)
    hist_path = out_dir / "hist.png"
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # MAE Polar Map
    fig, _ = plot_polar_map(df, ds, hemisphere=config.get("hemisphere", "north"))
    polar_path = out_dir / "polar_map.png"
    fig.savefig(polar_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return {
        "qq": str(qq_path),
        "hexbin": str(hex_path),
        "hist": str(hist_path),
        "polar_map": str(polar_path)
    }
