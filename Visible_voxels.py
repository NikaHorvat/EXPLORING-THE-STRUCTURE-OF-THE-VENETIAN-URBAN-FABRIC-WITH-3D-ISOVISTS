import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from joblib import Parallel, delayed

# parameters
REFERENCE_ID = 112666  # middle of square
# REFERENCE_ID = 314538  # narrow street
MAX_RADIUS  = 200.0
STEP_SIZE   = 1.0
VOXEL_SIZE  = 7.0
N_ANGLE_BINS = 360
OUTPUT_CSV  = "Visible_voxels.csv"

def voxel_coord(x, y, z):
    return (
        int(math.floor(x / VOXEL_SIZE)),
        int(math.floor(y / VOXEL_SIZE)),
        int(round(z / VOXEL_SIZE)),
    )

def build_occupied_set(df):
    occupied = set()
    for row in df.itertuples(index=False):
        ix, iy, iz = voxel_coord(row.center_x, row.center_y, row.center_z)
        occupied.add((ix, iy, iz))
    return occupied

def is_line_of_sight_clear(obs, tgt, occupied_voxels, step=STEP_SIZE):
    ox, oy, oz = obs.center_x, obs.center_y, obs.center_z
    tx, ty, tz = tgt.center_x, tgt.center_y, tgt.center_z

    dx = tx - ox
    dy = ty - oy
    dz = tz - oz
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1e-9:
        return False

    steps = int(dist // step)
    ux = dx / dist
    uy = dy / dist
    uz = dz / dist
    offset = 0.01

    for i in range(steps+1):
        d = offset + i * step
        if d > dist:
            d = dist
        x = ox + ux * d
        y = oy + uy * d
        z = oz + uz * d
        if d >= dist - 0.0001:
            return True
        # Check if this coordinate is in an occupied voxel
        vcoord = voxel_coord(x, y, z)
        # Skip the target's own voxel
        if vcoord == voxel_coord(tx, ty, tz):
            return True
        # If found a building/facade in the path => blocked
        if vcoord in occupied_voxels:
            return False
    return True

def main():
    # 1) Load the CSV
    df = pd.read_csv("3D_classification_plot.csv")

    # 2) Find observer voxel
    obs_df = df[df["cell_id"] == REFERENCE_ID]
    if obs_df.empty:
        raise ValueError(f"No voxel with cell_id={REFERENCE_ID} found!")
    observer = obs_df.iloc[0]
    ox, oy, oz = observer.center_x, observer.center_y, observer.center_z
    print(f"Observer voxel {REFERENCE_ID} at (X={ox}, Y={oy}, Z={oz})")

    # 3) Filter all voxels within 200m
    dx = df["center_x"] - ox
    dy = df["center_y"] - oy
    dist2d = np.sqrt(dx*dx + dy*dy)
    df_in_radius = df[dist2d <= MAX_RADIUS].copy()

    # 4) Build the set of occupied voxel coords
    blocking_classes = {"building", "facade"}
    blockers = df_in_radius[df_in_radius["classification"].isin(blocking_classes)].copy()

    if "uid" in df.columns and "uid" in observer:
        my_building_uid = observer.uid
        blockers = blockers[blockers["uid"] != my_building_uid]

    occupied_voxels = build_occupied_set(blockers)
    print(f"Number of blocking voxel coords: {len(occupied_voxels)}")

    # 5) Build occupied set
    occupied_voxels = build_occupied_set(blockers)

    # 6) Compute horizontal angle & distance for each candidate
    df = df_in_radius.copy()
    dx = df.center_x - ox
    dy = df.center_y - oy
    df['dist2d'] = np.hypot(dx, dy)

    angles = (np.arctan2(dy, dx) + 2*np.pi) % (2*np.pi)
    df['angle_bin'] = np.floor(angles / (2*np.pi) * N_ANGLE_BINS).astype(int)

    # 7) Sweep each ray
    df['visible'] = False
    for bin_id, ray in df.groupby('angle_bin'):

        ray = ray.sort_values('dist2d')
        blocked = False
        for idx, tgt in ray.iterrows():
            if blocked:
                continue

            if tgt.cell_id == REFERENCE_ID:
                df.at[idx, 'visible'] = True
                continue

            if is_line_of_sight_clear(observer, tgt, occupied_voxels, step=STEP_SIZE):
                df.at[idx, 'visible'] = True
            else:
                df.at[idx, 'visible'] = False
                blocked = True

    # 8) Export and plot
    df_visible = df[df.visible]
    df_visible.to_csv(OUTPUT_CSV, index=False)
    plot_3d(observer, df, df_visible)

def plot_3d(observer, df_in_radius, df_visible):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    df_nonvis = df_in_radius[~df_in_radius["visible"]]

    # Red = blocked
    if not df_nonvis.empty:
        ax.scatter(df_nonvis["center_x"], df_nonvis["center_y"], df_nonvis["center_z"],
                   c='red', s=5, marker='o', alpha=0.5, label='Non-visible')

    # Green = visible
    if not df_visible.empty:
        ax.scatter(df_visible["center_x"], df_visible["center_y"], df_visible["center_z"],
                   c='green', s=5, marker='o', alpha=0.5, label='Visible')

    # Blue = observer
    ax.scatter(observer.center_x, observer.center_y, observer.center_z,
               c='blue', s=60, marker='^', label='Observer')

    ax.set_xlabel("X coordinate (m)", labelpad=10, fontsize=12)
    ax.set_ylabel("Y coordinate (m)", labelpad=10, fontsize=12)
    ax.set_zlabel("Z coordinate (m)", labelpad=10, fontsize=12)

    plt.title("Voxel visibility", pad=20, fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()