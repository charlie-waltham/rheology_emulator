import io
import json
import logging
import pickle
import traceback
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import xarray as xr
import yaml
from matplotlib.figure import Figure
from PIL import Image
from scipy.interpolate import griddata
from scipy.spatial import KDTree

FIG_SIZE = (4, 4)
FIG_SIZE_LARGE = (6, 6)
WIDE_FIG_SIZE = (8, 4)

SEASON_NAMES = {
    "DJF": "December, January, February",
    "MAM": "March, April, May",
    "JJA": "June, July, August",
    "SON": "September, October, November",
}


def metrics(ds: xr.Dataset) -> dict:
    """
    Compute evaluation metrics between true and predicted values.
    It is assumed that labels are change in velocity, rather than the net velocity.
    """

    y_true = torch.tensor(ds.true.values)
    y_pred = torch.tensor(ds.pred.values)

    y_true_mag = torch.tensor(ds.true_magnitude.values)
    y_pred_mag = torch.tensor(ds.pred_magnitude.values)

    values = {}
    values["mean_true"] = torch.mean(y_true_mag)
    values["mean_pred"] = torch.mean(y_pred_mag)
    values["std"] = torch.std(y_pred_mag)
    values["mse"] = F.mse_loss(y_pred, y_true)
    values["mae"] = F.l1_loss(y_pred, y_true)
    values["rmse_cms"] = torch.sqrt(values["mse"]) * 100
    values["skill"] = 1 - values["mse"] / F.mse_loss(torch.zeros_like(y_true), y_true)

    true_dev = y_true_mag - values["mean_true"]
    pred_dev = y_pred_mag - values["mean_pred"]
    values["acc"] = torch.sum(true_dev * pred_dev) / torch.sqrt(
        torch.sum(true_dev**2) * torch.sum(pred_dev**2)
    )

    for key, value in values.items():
        values[key] = value.item()

    return values


def plot_qq(
    ds: xr.Dataset,
    quantiles: int = 200,
    xylim: float | None = None,
    ax: plt.Axes | None = None,
):
    """Create a QQ plot of predictions vs true values and return fig, ax."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    else:
        fig = ax.figure

    q = np.linspace(0.0, 1.0, quantiles)
    y_true = ds.true.values
    y_pred = ds.pred.values
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

    ax.set_xlabel("True quantiles")
    ax.set_ylabel("Pred quantiles")
    ax.set_title("QQ plot")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_hexbin(
    ds: xr.Dataset,
    gridsize: int = 60,
    extent: tuple[float, float, float, float] | None = None,
    ax: plt.Axes | None = None,
    title="Sea Ice Speed Hexbin True vs. Pred",
):
    """Create a hexbin scatter plot with log color scale."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    else:
        fig = ax.figure

    hb = ax.hexbin(
        ds.true_magnitude,
        ds.pred_magnitude,
        gridsize=gridsize,
        bins="log",
        cmap="viridis",
        extent=extent,
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("log10(N)")

    mn = 0 if extent is None else extent[0]
    mx = ds.pred_magnitude.max() if extent is None else extent[1]
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1, label="1:1")

    ax.set_xlabel("True Values")
    ax.set_ylabel("Predictions")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small")
    return fig, ax


def plot_hist(
    ds: xr.Dataset,
    bins: int = 80,
    x_range: tuple[float, float] | None = None,
    density: bool = True,
    ax: plt.Axes | None = None,
    title: str = "Sea Ice Speed True vs. Pred",
):
    """Plot normalized histograms of true and predicted values."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
    else:
        fig = ax.figure

    y_true = ds.true_magnitude.values
    y_pred = ds.pred_magnitude.values

    # determine plotting range: if user provided x_range, use it; otherwise
    # compute central 99.5% coverage (percentiles 0.25 and 99.75) of combined data
    if x_range is None:
        # combine true and pred into a single 1-D array; concatenating empty arrays yields empty array
        combined = np.concatenate([y_true.ravel(), y_pred.ravel()])

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

    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density" if density else "Count")
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
    vmin=1e-6,
    vmax=1,
    log=True,
    colourmap="viridis",
    title: str = "Sea Ice Velocity MAE",
    colourbar_label="Mean Absolute Error (m/s)",
):
    """Create a polar-projected map of values."""

    lat, lon = lat_lon[:, 0], lat_lon[:, 1]

    scalar = np.linalg.norm(values, axis=0) if values.ndim > 1 else values

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
        tree = KDTree(np.column_stack((x_points, y_points)))
        # Query tree (flatten grid for query)
        grid_pixels = np.column_stack((grid_x_2d.ravel(), grid_y_2d.ravel()))
        dist, _ = tree.query(grid_pixels)
        dist = dist.reshape(grid_x_2d.shape)

        grid[dist > dist_threshold] = np.nan
        return grid

    # Interpolate values
    grid_scalar = interpolate_and_mask(scalar)

    fig = plt.figure(figsize=FIG_SIZE_LARGE)
    ax = plt.axes(projection=projection, facecolor="cornflowerblue")
    ax.set_extent(extent, src_crs)
    ax.add_feature(cfeature.LAND, zorder=2, facecolor="grey", edgecolor="black")
    ax.gridlines()

    # Circular Boundary
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    # Define norm
    if log:
        norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # Plot scalars
    mesh = ax.pcolormesh(
        grid_x_2d,
        grid_y_2d,
        grid_scalar,
        transform=projection,
        norm=norm,
        cmap=colourmap,
        shading="auto",
        zorder=1,
    )
    plt.colorbar(mesh, ax=ax, label=colourbar_label)

    # Don't use for now as u and v don't necessarily correspond to lat and lon
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


def plot_polar_mae(ds: xr.Dataset, base_stride=1_000_000, **kwargs):
    stride = max(1, ds.sizes["indices"] // base_stride)

    # Extract Coordinates
    lat = ds.coords["lat"].values[::stride]
    lon = ds.coords["lon"].values[::stride]
    lat_lon = np.column_stack((lat, lon))

    y_pred = torch.tensor(ds.pred.values[:, ::stride])
    y_true = torch.tensor(ds.true.values[:, ::stride])

    mae = F.l1_loss(y_pred, y_true, reduction="none")
    mae = mae.mean(dim=0)

    return plot_polar_map(mae.numpy(), lat_lon, **kwargs)


def plot_polar_skill(ds: xr.Dataset, **kwargs):
    # Extract Coordinates
    lat = ds.coords["lat"].values
    lon = ds.coords["lon"].values
    lat_lon = np.column_stack((lat, lon))

    y_pred = torch.tensor(ds.pred.values)
    y_true = torch.tensor(ds.true.values)

    skill = 1 - F.mse_loss(y_pred, y_true, reduction="none") / F.mse_loss(
        torch.zeros_like(y_true), y_true, reduction="none"
    )
    skill = skill.mean(dim=0)

    return plot_polar_map(
        skill.numpy(),
        lat_lon,
        title="Sea Ice Velocity Prediction Skill",
        colourbar_label="Prediction Skill",
        vmin=0,
        vmax=1,
        log=False,
        **kwargs,
    )


def plot_to_buffer(fig: Figure, **kwargs):
    """https://github.com/paulgavrikov/parallel-matplotlib-grid/blob/main/parallelplot/plot.py"""
    buf = io.BytesIO()
    fig.savefig(buf, **kwargs)
    buf.seek(0)
    img = np.array(Image.open(buf))
    buf.close()
    plt.close(fig)
    return img


def _plot_season_task(season, ds_season, polar_kwargs):
    """Helper function for multiprocessing seasonal plots."""
    print(f"Processing {season}")
    try:
        plt.style.use("seaborn-v0_8")

        hist, _ = plot_hist(
            ds_season,
            x_range=(0, 0.05),
            density=False,
            title=season,
        )
        hexbin, _ = plot_hexbin(
            ds_season,
            extent=(0, 0.1, 0, 0.1),
            title=season,
        )
        polar_map, _ = plot_polar_mae(
            ds_season,
            title=season,
            **polar_kwargs,
        )

        figs = {
            "hist": plot_to_buffer(hist, dpi=600, bbox_inches="tight"),
            "hexbin": plot_to_buffer(hexbin, dpi=600, bbox_inches="tight"),
            "polar_map": plot_to_buffer(polar_map, dpi=600, bbox_inches="tight"),
        }
        plt.close("all")

        return figs
    except Exception:
        print(traceback.format_exc())


def plot_by_season(
    ds: xr.Dataset,
    out_dir: Path,
    polar_kwargs: dict = None,
):
    polar_kwargs = {} if polar_kwargs is None else polar_kwargs

    logging.info("Splitting data into seasons")

    # Compute seasons to memory to avoid dask grouping error
    seasons = ds["time_features"].dt.season.values
    ds = ds.assign_coords(season=("indices", seasons))
    groups = ds.groupby("season")

    # Don't bother if dataset only covers one season
    if len(groups) <= 1:
        return

    tasks = []
    for season, ds_group in groups:
        ds_season = ds_group.drop_vars("season")
        tasks.append((SEASON_NAMES.get(season), ds_season, polar_kwargs))

    # Use multiprocessing to plot
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(tasks)) as pool:
        month_figs = pool.starmap(_plot_season_task, tasks)

    hist_fig, hist_axs = plt.subplots(
        2, 2, figsize=(10, 10), gridspec_kw={"wspace": 0, "hspace": 0}
    )
    hexbin_fig, hexbin_axs = plt.subplots(
        2, 2, figsize=(10, 10), gridspec_kw={"wspace": 0, "hspace": 0}
    )
    polar_fig, polar_axs = plt.subplots(
        2, 2, figsize=(10, 10), gridspec_kw={"wspace": 0, "hspace": 0}
    )

    hist_fig.suptitle("Seasonal sea ice velocity Histogram True vs. Pred")
    hexbin_fig.suptitle("Seasonal sea ice velocity Hexbin True vs. Pred")
    polar_fig.suptitle("Seasonal sea ice velocity MAE")

    hist_axs = hist_axs.T.flatten()
    hexbin_axs = hexbin_axs.T.flatten()
    polar_axs = polar_axs.T.flatten()

    for i in range(len(month_figs)):
        figures = month_figs[i]
        hist_axs[i].grid(False)
        hist_axs[i].axis("off")
        hexbin_axs[i].grid(False)
        hexbin_axs[i].axis("off")
        polar_axs[i].grid(False)
        polar_axs[i].axis("off")

        hist_axs[i].imshow(figures["hist"])
        hexbin_axs[i].imshow(figures["hexbin"])
        polar_axs[i].imshow(figures["polar_map"])

    seasonal_dir = out_dir / "seasonal"
    seasonal_dir.mkdir(parents=True, exist_ok=True)

    hist_fig.savefig(seasonal_dir / "hist.png", dpi=600, bbox_inches="tight")
    hexbin_fig.savefig(seasonal_dir / "hexbin.png", dpi=600, bbox_inches="tight")
    polar_fig.savefig(seasonal_dir / "polar_map.png", dpi=1000, bbox_inches="tight")
    plt.close(hist_fig)
    plt.close(hexbin_fig)
    plt.close(polar_fig)


def attributions(args: dict, config: dict):
    with open(args["eval_path"] + "/attributions.pkl", "rb") as file:
        attributions: dict = pickle.load(file)

    label_map = {
        "sivelv": "$V_v$",
        "sivelu": "$V_u$",
    }
    feature_map = {
        "siconc": "SIC",
        "sithic": "SIT",
        "sivel": "$V$",
        "tau_ai": "$\\tau_{ai}$",
        "tau_oi": "$\\tau_{wi}$",
    }

    features = config["train_features"]

    # Find and group u/v vector pairs
    def get_base_name(feat):
        if feat.startswith("u") and feat.replace("u", "v", 1) in features:
            return feat[1:]
        if feat.startswith("v") and feat.replace("v", "u", 1) in features:
            return feat[1:]
        if feat.endswith("u") and feat[:-1] + "v" in features:
            return feat[:-1]
        if feat.endswith("v") and feat[:-1] + "u" in features:
            return feat[:-1]
        return feat

    # Map raw features to their grouped names and keep the order
    grouped_features = []
    feature_to_group_idx = {}
    for feat in features:
        base = get_base_name(feat)
        if base not in grouped_features:
            grouped_features.append(base)
        feature_to_group_idx[feat] = grouped_features.index(base)

    x = np.arange(len(grouped_features))
    width = 1.0 / (len(config["train_labels"]) + 1)

    fig, ax = plt.subplots(figsize=FIG_SIZE)

    for key, val in attributions.items():
        raw_mean_vals = np.abs(np.mean(val, axis=0))

        # Sum U and V components into the grouped array
        grouped_vals = np.zeros(len(grouped_features))
        for i, feat in enumerate(features):
            idx = feature_to_group_idx[feat]
            grouped_vals[idx] += raw_mean_vals[i]

        target_name = config["train_labels"][key]
        ax.bar(
            x + width * key,
            grouped_vals,
            width=width,
            label=label_map.get(target_name, target_name),
        )

    ax.legend()
    ax.set_xticks(x + width * (len(config["train_labels"]) - 1) / 2)
    ax.set_xticklabels(feature_map.get(f, f) for f in grouped_features)
    ax.set_title("Absolute Relative Feature Importances")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Absolute Importance")

    return fig, ax


def evaluate_and_save(args: dict):
    """Load data, plot QQ/hexbin/hist etc., and save figures to results_dir."""

    out_dir = Path(args["eval_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_ds = xr.open_dataset(args["eval_path"] + "/ytrue_ypred_test.cdf")

    # Retrieve used dataset
    with open(out_dir / "used_training_config.yaml") as file:
        config = yaml.safe_load(file)
    train_ds = xr.open_zarr(config["pairs_path"])
    train_ds = train_ds.drop_vars(["features", "labels", "d_labels"], errors="ignore")
    indices = pred_ds.coords["indices"]
    train_ds = train_ds.isel(z=indices)

    ds = xr.combine_by_coords([train_ds, pred_ds])

    # Metrics
    metrics_results = metrics(ds)
    logging.info("Metrics:")
    for key, value in metrics_results.items():
        logging.info(f"    {key}: {value}")

    with open(out_dir / "metrics.json", "w") as f:
        f.write(json.dumps(metrics_results, indent=4))

    # QQ
    logging.info("qq")
    fig, _ = plot_qq(ds)
    qq_path = out_dir / "qq.png"
    fig.savefig(qq_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Hexbin
    logging.info("hexbin")
    fig, _ = plot_hexbin(ds, extent=[0, 0.1, 0, 0.1])
    hex_path = out_dir / "hexbin.png"
    fig.savefig(hex_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Histogram
    logging.info("histogram")
    fig, _ = plot_hist(ds, density=False)
    hist_path = out_dir / "hist.png"
    fig.savefig(hist_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Attributions
    logging.info("attributions")
    fig, _ = attributions(args, config)
    attributions_path = out_dir / "attributions.png"
    fig.savefig(attributions_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Polar Maps
    logging.info("error polar map")
    fig, _ = plot_polar_mae(ds, hemisphere=config.get("hemisphere", "north"))
    polar_path = out_dir / "polar_map_error.png"
    fig.savefig(polar_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    logging.info("skill polar map")
    fig, _ = plot_polar_skill(ds, hemisphere=config.get("hemisphere", "north"))
    polar_path = out_dir / "polar_map_skill.png"
    fig.savefig(polar_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    # Seasonal plots
    plot_by_season(
        ds, out_dir, polar_kwargs={"hemisphere": config.get("hemisphere", "north")}
    )
