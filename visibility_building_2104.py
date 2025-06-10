import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# parameters
INPUT_BUILDING_VOXELS_CSV = "3D_buildings_around_2104.csv"
INPUT_2D_CSV             = "2D_classification_around_2104.csv"
TARGET_EDIFI_ID          = "2104"
MAX_RADIUS               = 20.0
STEP_SIZE                = 1.0
VOXEL_SIZE               = 1.0
N_ANGLE_BINS             = 360
OUTPUT_COUNTS_CSV        = "Visible_counts_2104_filled.csv"

def voxel_coord(x: float, y: float, z: float) -> tuple[int,int,int]:
    return (
        int(math.floor(x / VOXEL_SIZE)),
        int(math.floor(y / VOXEL_SIZE)),
        int(round(z / VOXEL_SIZE))
    )

def build_occupied_set(df_building: pd.DataFrame) -> set[tuple[int,int,int]]:
    occ = set()
    for row in df_building.itertuples(index=False):
        occ.add(voxel_coord(row.x_center, row.y_center, row.z_center))
    return occ

def is_line_of_sight_clear(
    obs: pd.Series,
    tgt: pd.Series,
    occupied: set[tuple[int,int,int]],
    step: float = STEP_SIZE
) -> bool:
    ox, oy, oz = obs.x_center, obs.y_center, obs.z_center
    tx, ty, tz = tgt.x_center, tgt.y_center, tgt.z_center

    dx, dy, dz = tx - ox, ty - oy, tz - oz
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1e-9:
        return True

    ux, uy, uz = dx/dist, dy/dist, dz/dist
    target_idx = voxel_coord(tx, ty, tz)
    offset = 1e-3
    steps = int(dist // step)

    for i in range(steps + 1):
        d_along = min(offset + i * step, dist)
        cx = ox + ux*d_along
        cy = oy + uy*d_along
        cz = oz + uz*d_along
        current_idx = voxel_coord(cx, cy, cz)
        if current_idx == target_idx:
            return True
        if current_idx in occupied:
            return False
    return True

def cube_faces(cx: float, cy: float, cz: float, size: float):
    half = size / 2.0
    corners = [
        (cx-half, cy-half, cz-half),
        (cx+half, cy-half, cz-half),
        (cx+half, cy+half, cz-half),
        (cx-half, cy+half, cz-half),
        (cx-half, cy-half, cz+half),
        (cx+half, cy-half, cz+half),
        (cx+half, cy+half, cz+half),
        (cx-half, cy+half, cz+half),
    ]
    return [
        [corners[0], corners[1], corners[2], corners[3]],
        [corners[4], corners[5], corners[6], corners[7]],
        [corners[0], corners[1], corners[5], corners[4]],
        [corners[3], corners[2], corners[6], corners[7]],
        [corners[1], corners[2], corners[6], corners[5]],
        [corners[0], corners[3], corners[7], corners[4]],
    ]

def main():
    # 1) Load building voxels
    df_building = pd.read_csv(INPUT_BUILDING_VOXELS_CSV, low_memory=False)
    if df_building.empty:
        raise RuntimeError(f"'{INPUT_BUILDING_VOXELS_CSV}' not found or empty.")
    print(f"Loaded {len(df_building)} building voxels.")

    # 2) Filter for target building (EDIFI_ID = 2104)
    df_2104 = df_building[df_building["EDIFI_ID"].astype(str) == TARGET_EDIFI_ID].copy()
    if df_2104.empty:
        raise RuntimeError(f"No voxels found for EDIFI_ID = {TARGET_EDIFI_ID}.")
    print(f"Found {len(df_2104)} voxels for building {TARGET_EDIFI_ID}.")

    # 3) Compute voxel coordinates (i, j, k) from (x_center, y_center, z_center)
    df_2104["i"] = (df_2104["x_center"] / VOXEL_SIZE).apply(math.floor).astype(int)
    df_2104["j"] = (df_2104["y_center"] / VOXEL_SIZE).apply(math.floor).astype(int)
    df_2104["k"] = (df_2104["z_center"] / VOXEL_SIZE).round().astype(int)

    # 4) Build a set of voxel coordinates
    coords_2104 = set(zip(df_2104["i"], df_2104["j"], df_2104["k"]))

    is_ext = []
    for row in df_2104.itertuples(index=False):
        i, j, k = row.i, row.j, row.k
        neighbors_six = [
            (i+1, j,   k),
            (i-1, j,   k),
            (i,   j+1, k),
            (i,   j-1, k),
            (i,   j,   k+1),
            (i,   j,   k-1)
        ]
        fully_enclosed = all(n in coords_2104 for n in neighbors_six)
        is_ext.append(not fully_enclosed)

    df_2104["is_exterior"] = is_ext
    df_exterior = df_2104[df_2104["is_exterior"]].copy()
    print(f"{len(df_exterior)} voxels remain on the outside (exterior).")

    # 5) Build occupied set from exterior voxels
    occupied = build_occupied_set(df_building)
    print(f"Occupied set contains {len(occupied)} total building indices.")

    # 6) Load 2D classification CSV
    df2d = pd.read_csv(INPUT_2D_CSV, low_memory=False)
    if df2d.empty:
        raise RuntimeError(f"'{INPUT_2D_CSV}' missing or empty.")
    df_targets_2d = df2d[df2d["classification"].isin(["street", "canal"])].copy()
    df_targets = pd.DataFrame({
        "x_center": df_targets_2d["x_center"],
        "y_center": df_targets_2d["y_center"],
        "z_center": np.zeros(len(df_targets_2d), dtype=float),
    })
    print(f"Found {len(df_targets)} target voxels (street+canal).")

    # 7) Build KD-Tree for target voxels
    target_xy = df_targets[["x_center", "y_center"]].to_numpy()
    tree_tgt = cKDTree(target_xy)

    # 8) Compute visibility counts for each exterior voxel
    visible_counts = []
    print("Computing visibility counts for each exterior 2104 voxel…")

    for obs_idx, obs_row in df_exterior.iterrows():
        ox, oy, oz = obs_row.x_center, obs_row.y_center, obs_row.z_center

        neighbor_idxs = tree_tgt.query_ball_point([ox, oy], r=MAX_RADIUS)
        if not neighbor_idxs:
            visible_counts.append(0)
            continue

        df_cands = df_targets.iloc[neighbor_idxs].copy()
        dx = df_cands["x_center"].to_numpy() - ox
        dy = df_cands["y_center"].to_numpy() - oy
        dist2d = np.hypot(dx, dy)
        df_cands["dist2d"] = dist2d

        angles = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
        angle_bins = (angles / (2*np.pi) * N_ANGLE_BINS).astype(int)
        df_cands["angle_bin"] = angle_bins

        df_cands = df_cands[df_cands["dist2d"] <= MAX_RADIUS + 1e-6].copy()
        if df_cands.empty:
            visible_counts.append(0)
            continue

        df_cands["visible"] = False

        for _, ray in df_cands.groupby("angle_bin"):
            ray_sorted = ray.sort_values("dist2d")
            blocked = False
            for idx, tgt in ray_sorted.iterrows():
                if blocked:
                    continue
                if is_line_of_sight_clear(obs_row, tgt, occupied, step=STEP_SIZE):
                    df_cands.at[idx, "visible"] = True
                else:
                    df_cands.at[idx, "visible"] = False
                    blocked = True

        count_vis = int(df_cands["visible"].sum())
        visible_counts.append(count_vis)

    df_exterior = df_exterior.reset_index(drop=True)
    df_exterior["visible_count"] = visible_counts

    # 9) Fill
    df_exterior.sort_values(["x_center", "y_center", "z_center"], ascending=[True, True, False], inplace=True)
    df_exterior["filled_count"] = df_exterior["visible_count"].copy()

    for (x, y), group in df_exterior.groupby(["x_center", "y_center"]):
        counts = group["filled_count"].to_numpy()
        for i in range(len(counts) - 1):
            if counts[i+1] == 0:
                counts[i+1] = counts[i]
        df_exterior.loc[group.index, "filled_count"] = counts

    # 10) Export
    df_export = df_exterior[["x_center", "y_center", "z_center", "filled_count"]].copy()
    df_export.rename(columns={"filled_count": "visible_count"}, inplace=True)
    df_export.to_csv(OUTPUT_COUNTS_CSV, index=False)
    print(f"Exported {len(df_export)} rows to '{OUTPUT_COUNTS_CSV}'.")

    # 11) Plot
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111, projection="3d")

    counts = df_export["visible_count"].to_numpy()
    norm = plt.Normalize(vmin=counts.min(), vmax=counts.max())
    cmap = plt.cm.get_cmap("hot")

    all_faces = []
    all_facecolors = []
    for _, row in df_export.iterrows():
        cx, cy, cz = row.x_center, row.y_center, row.z_center
        vc = row.visible_count
        facecolor = cmap(norm(vc))
        faces = cube_faces(cx, cy, cz, VOXEL_SIZE)
        all_faces.extend(faces)
        all_facecolors.extend([facecolor] * 6)

    poly_collection = Poly3DCollection(
        all_faces,
        facecolors=all_facecolors,
        edgecolors="k",
        linewidths=0.05,
        alpha=1.0
    )
    ax.add_collection3d(poly_collection)

    xs = df_export["x_center"]
    ys = df_export["y_center"]
    zs = df_export["z_center"]
    pad = VOXEL_SIZE
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_zlim(zs.min() - pad, zs.max() + pad)

    ax.set_box_aspect((1,1,1))
    ax.set_xlabel("X (m)", labelpad=10, fontsize=12)
    ax.set_ylabel("Y (m)", labelpad=10, fontsize=12)
    ax.set_zlabel("Z (m)", labelpad=10, fontsize=12)
    ax.set_title("2104 Exterior Voxels (Filled) Visible Street/Canal Count", fontsize=14)

    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array(counts)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Number of street/canal voxels visible", fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()