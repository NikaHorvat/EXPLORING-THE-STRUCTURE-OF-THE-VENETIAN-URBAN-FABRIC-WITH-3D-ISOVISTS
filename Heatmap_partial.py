import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

# Parameters
REFERENCE_ID = 314538
MAX_RADIUS   = 200.0
STEP_SIZE    = 1.0
VOXEL_SIZE   = 7.0
OUTPUT_CSV   = "Visibility_heatmap.csv"

def voxel_coord(x, y, z):
    return (
        int(math.floor(x / VOXEL_SIZE)),
        int(math.floor(y / VOXEL_SIZE)),
        int(round    (z / VOXEL_SIZE)),
    )
    
# Builds a set of occupied voxel coordinates from the DataFrame
def build_occupied_set(df):
    occ = set()
    for row in df.itertuples(index=False):
        occ.add(voxel_coord(row.center_x, row.center_y, row.center_z))
    return occ

# Checks if the line of sight from observer to target is clear
def is_line_of_sight_clear(obs, tgt, occupied, step=STEP_SIZE):
    ox, oy, oz = obs.center_x, obs.center_y, obs.center_z
    tx, ty, tz = tgt.center_x, tgt.center_y, tgt.center_z
    dx, dy, dz = tx-ox, ty-oy, tz-oz
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1e-6:
        return False
    ux, uy, uz = dx/dist, dy/dist, dz/dist
    offset = 0.01
    target_idx = voxel_coord(tx, ty, tz)
    steps = int(dist // step)
    for i in range(steps+1):
        d = min(offset + i*step, dist)
        x, y, z = ox + ux*d, oy + uy*d, oz + uz*d
        if d >= dist - 1e-6:
            return True
        vid = voxel_coord(x, y, z)
        if vid == target_idx:
            return True
        if vid in occupied:
            return False
    return True

def cube_faces(cx, cy, cz):
    hs = VOXEL_SIZE/2.0
    v = [
        (cx-hs, cy-hs, cz-hs), (cx+hs, cy-hs, cz-hs),
        (cx+hs, cy+hs, cz-hs), (cx-hs, cy+hs, cz-hs),
        (cx-hs, cy-hs, cz+hs), (cx+hs, cy-hs, cz+hs),
        (cx+hs, cy+hs, cz+hs), (cx-hs, cy+hs, cz+hs),
    ]
    return [
        [v[0],v[1],v[2],v[3]],  # bottom
        [v[4],v[5],v[6],v[7]],  # top
        [v[0],v[1],v[5],v[4]],  # front
        [v[2],v[3],v[7],v[6]],  # back
        [v[1],v[2],v[6],v[5]],  # right
        [v[4],v[7],v[3],v[0]],  # left
    ]

def main():
    # 1) Load and filter to 200m
    df = pd.read_csv("3d_classification_plot.csv")
    center = df[df.cell_id == REFERENCE_ID]
    if center.empty:
        raise ValueError("Reference ID not found.")
    center = center.iloc[0]
    dx = df.center_x - center.center_x
    dy = df.center_y - center.center_y
    df_r = df[np.hypot(dx, dy) <= MAX_RADIUS].copy()

    # 2) Blockers
    blockers = df_r[df_r.classification.isin(["building", "facade"])]
    if 'uid' in blockers.columns:
        blockers = blockers[blockers.uid != center.uid]
    occupied = build_occupied_set(blockers)

    # 3) Observers are the street voxels
    observers = df_r[df_r.classification == "street"].reset_index(drop=True)
    n_obs = len(observers)
    print(f"Using {n_obs} street voxels as observers")

    # 4) Compute vis_count
    df_r['vis_count'] = 0
    def count_for(obs_row):
        return [1 if is_line_of_sight_clear(obs_row, tgt, occupied) else 0
                for _, tgt in df_r.iterrows()]

    all_counts = Parallel(n_jobs=-1)(
        delayed(count_for)(observers.iloc[i]) for i in range(n_obs)
    )
    # sum up per-target
    df_r['vis_count'] = np.sum(all_counts, axis=0)
    df_r['vis_frac'] = df_r['vis_count'] / n_obs

    # 5) Save results to CSV
    df_r.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported counts to {OUTPUT_CSV}")

    # 6) Build cube geometry
    cmap = plt.get_cmap("inferno")
    max_count = df_r['vis_count'].max()
    norm = plt.Normalize(vmin=0, vmax=max_count)

    faces_all = []
    colors_all = []
    for _, row in df_r.iterrows():
        color = cmap(norm(row.vis_count))
        for face in cube_faces(row.center_x, row.center_y, row.center_z):
            faces_all.append(face)
            colors_all.append(color)

    # 7) Plot
    fig = plt.figure(figsize=(12,10))
    ax  = fig.add_subplot(111, projection='3d')
    poly = Poly3DCollection(faces_all, facecolors=colors_all,
                            edgecolors='none', linewidths=0, alpha=0.9)
    ax.add_collection3d(poly)

    pad = VOXEL_SIZE
    ax.set_xlim(df_r.center_x.min()-pad, df_r.center_x.max()+pad)
    ax.set_ylim(df_r.center_y.min()-pad, df_r.center_y.max()+pad)
    ax.set_zlim(df_r.center_z.min()-pad, df_r.center_z.max()+pad)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Visibility count", rotation=270, labelpad=15)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D Voxel Heatmap (visibility count)")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    main()