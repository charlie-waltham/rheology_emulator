import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.colors as colors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
from scipy.spatial import cKDTree
import yaml
import torch
import torch.nn.functional as F

TRUE_COL = "true_sivelu"
PRED_COL = "pred_sivelu"
TRUE_COL_V = "true_sivelv"
PRED_COL_V = "pred_sivelv"
INDICES_COL = "indices"


def load_df(
    csv_path: str, true_col: str = TRUE_COL, pred_col: str = PRED_COL
) -> pd.DataFrame:
    """Load a CSV and ensure it contains the expected columns."""
    df = pd.read_csv(csv_path)
    missing = [col for col in (true_col, pred_col) if col not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df


def metrics(df: pd.DataFrame) -> dict:
    """
    Compute evaluation metrics between true and predicted values.
    It is assumed that labels are change in velocity, rather than the net velocity.
    """

    # Compute vector magnitudes
    y_true = torch.hypot(torch.tensor(df[TRUE_COL]), torch.tensor(df[TRUE_COL_V]))
    y_pred = torch.hypot(torch.tensor(df[PRED_COL]), torch.tensor(df[PRED_COL_V]))

    values = {}
    values["mse"] = F.mse_loss(y_pred, y_true)
    values["mae"] = F.l1_loss(y_pred, y_true)
    values["rmse_cms"] = torch.sqrt(values["mse"]) * 100
    values["skill"] = 1 - values["mse"] / F.mse_loss(torch.zeros_like(y_true), y_true)

    true_dev = y_true - torch.mean(y_true)
    pred_dev = y_pred - torch.mean(y_pred)
    values["acc"] = torch.sum(true_dev * pred_dev) / torch.sqrt(
        torch.sum(true_dev**2) * torch.sum(pred_dev**2)
    )

    for key, value in values.items():
        values[key] = value.item()

    return values


def plot_qq(
    df: pd.DataFrame,
    true_col: str = TRUE_COL,
    pred_col: str = PRED_COL,
    quantiles: int = 200,
    xylim: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
):
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
        mae = float("nan")
        bias = float("nan")

    ax.plot(
        q_true, q_pred, ".", alpha=0.6, label=f"QQ (MAE={mae:.1e}, bias={bias:.1e})"
    )

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


def plot_hexbin(
    df: pd.DataFrame,
    true_col: str = TRUE_COL,
    pred_col: str = PRED_COL,
    gridsize: int = 60,
    extent: Optional[Tuple[float, float, float, float]] = None,
    ax: Optional[plt.Axes] = None,
):
    """Create a hexbin scatter plot with log color scale."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure

    hb = ax.hexbin(
        df[true_col],
        df[pred_col],
        gridsize=gridsize,
        bins="log",
        cmap="viridis",
        extent=extent,
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(N)")

    mn = df[[true_col, pred_col]].min().min() if extent is None else extent[0]
    mx = df[[true_col, pred_col]].max().max() if extent is None else extent[1]
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="1:1")

    fig.set_size_inches(10, 10)
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predictions")
    ax.set_title("Hexbin (Pred vs True)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_hist(
    df: pd.DataFrame,
    true_col: str = TRUE_COL,
    pred_col: str = PRED_COL,
    bins: int = 80,
    x_range: Optional[Tuple[float, float]] = None,
    density: bool = True,
    ax: Optional[plt.Axes] = None,
):
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

    ax.hist(
        y_true,
        bins=bins,
        range=x_range_use,
        density=density,
        alpha=0.6,
        label="True",
        color="C0",
    )
    ax.hist(
        y_pred,
        bins=bins,
        range=x_range_use,
        density=density,
        alpha=0.6,
        label="Pred",
        color="C1",
    )

    fig.set_size_inches(10, 10)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title("Histogram (True vs Pred)")
    ax.set_yscale("log")
    if x_range_use is not None:
        ax.set_xlim(x_range_use)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_polar_map(
    values: np.ndarray,
    lat_lon: np.ndarray,
    quiver: bool = False,
    quiver_stride: int = 32,
    hemisphere: str = "north",
    resolution: int = 1024,
    lat_cutoff: int = 50,
    dist_threshold: int = 25000,
    title: str = "Sea Ice Velocity MAE",
):
    """Create a polar-projected map of values."""

    lat, lon = lat_lon[:, 0], lat_lon[:, 1]

    if values.ndim > 1:
        scalar = np.hypot(values[:, 0], values[:, 1])
    else:
        scalar = values

    if hemisphere == "south":
        projection = ccrs.SouthPolarStereo()
        extent = [-180, 180, -lat_cutoff, -90]
    else:
        projection = ccrs.NorthPolarStereo()
        extent = [-180, 180, lat_cutoff, 90]

    src_crs = ccrs.PlateCarree()

    # Project lat/lon to metres
    coords_proj = projection.transform_points(src_crs, lon, lat)
    x_points = coords_proj[:, 0]
    y_points = coords_proj[:, 1]

    grid_x = np.linspace(-4000000, 4000000, resolution)
    grid_y = np.linspace(-4000000, 4000000, resolution)
    grid_x_2d, grid_y_2d = np.meshgrid(grid_x, grid_y)

    def interpolate_and_mask(values):
        # Linear interpolation
        grid = griddata(
            (x_points, y_points), values, (grid_x_2d, grid_y_2d), method="linear"
        )
        # Distance masking
        tree = cKDTree(np.column_stack((x_points, y_points)))
        # Query tree (flatten grid for query)
        grid_pixels = np.column_stack((grid_x_2d.ravel(), grid_y_2d.ravel()))
        dist, _ = tree.query(grid_pixels)
        dist = dist.reshape(grid_x_2d.shape)

        grid[dist > dist_threshold] = np.nan
        return grid

    # Interpolate values
    grid_scalar = interpolate_and_mask(scalar)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=projection)
    ax.set_extent(extent, src_crs)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor="gray", edgecolor="black")
    ax.gridlines()

    # Circular Boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Plot scalars
    mesh = ax.pcolormesh(
        grid_x_2d,
        grid_y_2d,
        grid_scalar,
        transform=projection,
        norm=colors.LogNorm(1e-6, 1),
        cmap="viridis",
        shading="auto",
        zorder=1,
    )
    plt.colorbar(mesh, ax=ax, label="Mean Absolute Error (m/s)")

    if quiver and values.ndim > 1:
        grid_u = interpolate_and_mask(values[:, 0])
        grid_v = interpolate_and_mask(values[:, 1])

        # Subsample grid for quiver plot
        quiver_slice = (
            slice(None, None, quiver_stride),
            slice(None, None, quiver_stride),
        )
        quiver_x = grid_x_2d[quiver_slice]
        quiver_y = grid_y_2d[quiver_slice]
        quiver_u = grid_u[quiver_slice]
        quiver_v = grid_v[quiver_slice]
        quiver_speed = np.hypot(quiver_u, quiver_v)

        quiver_u_norm = quiver_u / quiver_speed
        quiver_v_norm = quiver_v / quiver_speed

        # Unproject x/y coords back to lat/lon for accurate directions
        quiver_geo = src_crs.transform_points(projection, quiver_x, quiver_y)
        quiver_lon = quiver_geo[:, :, 0]
        quiver_lat = quiver_geo[:, :, 1]

        # Plot vectors
        q = ax.quiver(
            quiver_lon,
            quiver_lat,
            quiver_u_norm,
            quiver_v_norm,
            quiver_speed,
            transform=ccrs.PlateCarree(),
            cmap="autumn",
            norm=colors.LogNorm(1e-5, 1e-1),
            scale=50,
            width=0.002,
            headwidth=3,
            zorder=3,
        )
        plt.colorbar(q, ax=ax, label="Vector Mean Absolute Error (m/s)")

    ax.set_title(title)
    return fig, ax


def plot_polar_sivelu(
    df: pd.DataFrame, ds: xr.Dataset, base_stride=10_000_000, **kwargs
):
    stride = max(1, len(df) // base_stride)

    # Extract Coordinates
    lat = ds.coords["lat"].values[::stride]
    lon = ds.coords["lon"].values[::stride]
    lat_lon = np.column_stack((lat, lon))

    # Extract Vector Components (and subset them)
    u_true = df[TRUE_COL].values[::stride]
    u_pred = df[PRED_COL].values[::stride]
    mae = np.abs(u_true - u_pred)

    return plot_polar_map(
        mae, lat_lon, title="Sea Ice Velocity U component MAE", **kwargs
    )


def plot_polar_sivelv(
    df: pd.DataFrame, ds: xr.Dataset, base_stride=10_000_000, **kwargs
):
    stride = max(1, len(df) // base_stride)

    # Extract Coordinates
    lat = ds.coords["lat"].values[::stride]
    lon = ds.coords["lon"].values[::stride]
    lat_lon = np.column_stack((lat, lon))

    # Extract Vector Components (and subset them)
    v_true = df[TRUE_COL_V].values[::stride]
    v_pred = df[PRED_COL_V].values[::stride]
    mae = np.abs(v_true - v_pred)

    return plot_polar_map(
        mae, lat_lon, title="Sea Ice Velocity V component MAE", **kwargs
    )


def plot_polar_vectors(
    df: pd.DataFrame, ds: xr.Dataset, base_stride=10_000_000, **kwargs
):
    stride = max(1, len(df) // base_stride)

    # Extract Coordinates
    lat = ds.coords["lat"].values[::stride]
    lon = ds.coords["lon"].values[::stride]
    lat_lon = np.column_stack((lat, lon))

    # Extract Vector Components (and subset them)
    u_true = df[TRUE_COL].values[::stride]
    u_pred = df[PRED_COL].values[::stride]
    v_true = df[TRUE_COL_V].values[::stride]
    v_pred = df[PRED_COL_V].values[::stride]

    # Technically not MAE because not absolute
    error = np.column_stack((u_true - u_pred, v_true - v_pred))

    return plot_polar_map(error, lat_lon, quiver=True, **kwargs)


def evaluate_and_save(args: dict):
    """Load data, plot QQ/hexbin/hist, and save figures to results_dir."""
    df = load_df(args["csv_path"])

    out_dir = Path(args["eval_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Retrieve used dataset
    with open(out_dir / "used_training_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    ds = xr.open_zarr(config["pairs_path"])
    indices = df[INDICES_COL].to_numpy()
    ds = ds.isel(z=indices)

    # Metrics
    metrics_results = metrics(df)
    print("Metrics:")
    for key, value in metrics_results.items():
        print(f"    {key}: {value}")

    with open(out_dir / "metrics.json", "w") as f:
        f.write(json.dumps(metrics_results, indent=4))

    # QQ
    print("qq")
    fig, _ = plot_qq(df)
    qq_path = out_dir / "qq.png"
    fig.savefig(qq_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Hexbin
    print("hexbin")
    fig, _ = plot_hexbin(df, extent=[-0.1, 0.1, -0.1, 0.1])
    hex_path = out_dir / "hexbin.png"
    fig.savefig(hex_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Histogram
    print("histogram")
    fig, _ = plot_hist(df, density=False)
    hist_path = out_dir / "hist.png"
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # MAE Polar Maps

    print("sivelu polar map")
    fig, _ = plot_polar_sivelu(df, ds, hemisphere=config.get("hemisphere", "north"))
    polar_path = out_dir / "polar_map_u.png"
    fig.savefig(polar_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("sivelv polar map")
    fig, _ = plot_polar_sivelv(df, ds, hemisphere=config.get("hemisphere", "north"))
    polar_path = out_dir / "polar_map_v.png"
    fig.savefig(polar_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("sivel polar map")
    fig, _ = plot_polar_vectors(df, ds, hemisphere=config.get("hemisphere", "north"))
    polar_path = out_dir / "polar_map.png"
    fig.savefig(polar_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
