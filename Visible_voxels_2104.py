import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# parameters
REFERENCE_ID = 217893
MAX_RADIUS   = 200.0
STEP_SIZE    = 1.0
VOXEL_SIZE   = 1.0
N_ANGLE_BINS = 360
OUTPUT_CSV   = "Visible_voxels_cube_2104.csv"

def voxel_coord(x, y, z):
    return (
        int(math.floor(x / VOXEL_SIZE)),
        int(math.floor(y / VOXEL_SIZE)),
        int(round(z / VOXEL_SIZE)),
    )


def build_occupied_set(df):
    occ = set()
    for row in df.itertuples(index=False):
        occ.add(voxel_coord(row.x_center, row.y_center, row.z_center))
    return occ

def is_line_of_sight_clear(obs, tgt, occupied, step=STEP_SIZE):
    ox, oy, oz = obs.x_center, obs.y_center, obs.z_center
    tx, ty, tz = tgt.x_center, tgt.y_center, tgt.z_center
    dx, dy, dz = tx - ox, ty - oy, tz - oz
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist < 1e-9:
        return False

    ux, uy, uz = dx / dist, dy / dist, dz / dist
    offset = 0.01
    target_v = voxel_coord(tx, ty, tz)
    steps = int(dist // step)

    for i in range(steps + 1):
        d = min(offset + i*step, dist)
        x, y, z = ox + ux*d, oy + uy*d, oz + uz*d

        if d >= dist - 1e-6:
            return True

        v = voxel_coord(x, y, z)
        if v == target_v:
            return True
        if v in occupied:
            return False

    return True


def cube_faces(cx, cy, cz, size):
    hs = size / 2.0
    v = [
        (cx - hs, cy - hs, cz - hs),
        (cx + hs, cy - hs, cz - hs),
        (cx + hs, cy + hs, cz - hs),
        (cx - hs, cy + hs, cz - hs),
        (cx - hs, cy - hs, cz + hs),
        (cx + hs, cy - hs, cz + hs),
        (cx + hs, cy + hs, cz + hs),
        (cx - hs, cy + hs, cz + hs),
    ]
    return [
        [v[0], v[1], v[2], v[3]],
        [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]],
        [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]],
        [v[4], v[7], v[3], v[0]],
    ]


def draw_voxels(ax, df, color, alpha=0.2):
    all_faces = []
    for _, row in df.iterrows():
        faces = cube_faces(row.x_center, row.y_center, row.z_center, VOXEL_SIZE)
        all_faces.extend(faces)

    poly = Poly3DCollection(
        all_faces,
        facecolors=color,
        edgecolors='k',
        linewidths=0.02,
        alpha=alpha
    )
    ax.add_collection3d(poly)


def plot_3d(obs, df_all, df_vis):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw non‐visible in red
    df_non = df_all[~df_all.visible]
    if not df_non.empty:
        draw_voxels(ax, df_non, color=(1, 0, 0), alpha=0.1)

    # Draw visible in green
    if not df_vis.empty:
        draw_voxels(ax, df_vis, color=(0, 1, 0), alpha=0.2)

    # Draw observer in blue
    obs_faces = cube_faces(obs.x_center, obs.y_center, obs.z_center, VOXEL_SIZE)
    ax.add_collection3d(Poly3DCollection(
        obs_faces,
        facecolors=(0, 0, 1),
        edgecolors=(0, 0, 1),
        linewidths=0,
        alpha=1.0
    ))

    ax.set_box_aspect((1, 1, 1))

    ax.set_xlim(df_all.x_center.min(), df_all.x_center.max())
    ax.set_ylim(df_all.y_center.min(), df_all.y_center.max())
    ax.set_zlim(df_all.z_center.min(), df_all.z_center.max())

    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_zlabel("Z coordinate (m)")
    red_patch = plt.Line2D(
        [0], [0], marker='s', color='w',
        label='Non-visible', markerfacecolor='r',
        markersize=10, alpha=0.5
    )
    green_patch = plt.Line2D(
        [0], [0], marker='s', color='w',
        label='Visible', markerfacecolor='g',
        markersize=10, alpha=0.5
    )
    blue_patch = plt.Line2D(
        [0], [0], marker='s', color='w',
        label='Observer', markerfacecolor='b',
        markersize=10, alpha=1.0
    )
    ax.legend(handles=[red_patch, green_patch, blue_patch], loc='upper left')

    plt.title("Voxel visibility")
    plt.tight_layout()
    plt.show()


def main():
    # 1) Load CSV
    df = pd.read_csv("3D_classification_ext_around_2104.csv")
    if df.empty:
        print("CSV is empty or not found.")
        return

    # 2) Find observer row by cell_id
    obs_df = df[df.cell_id == REFERENCE_ID]
    if obs_df.empty:
        raise ValueError(f"No voxel with cell_id={REFERENCE_ID} found!")
    obs = obs_df.iloc[0]

    # 3) Filter to within MAX_RADIUS (in the XY plane)
    dx = df.x_center - obs.x_center
    dy = df.y_center - obs.y_center
    df_r = df[np.hypot(dx, dy) <= MAX_RADIUS].copy()

    # 4) Identify “blockers”
    blockers = df_r[df_r.classification.isin(["building", "facade"])].copy()
    if "uid" in df.columns:
        blockers = blockers[blockers.uid != obs.uid]
    occupied = build_occupied_set(blockers)

    # 5) Compute 2D distance & angle bin for each candidate
    dx2 = df_r.x_center - obs.x_center
    dy2 = df_r.y_center - obs.y_center
    df_r['dist2d'] = np.hypot(dx2, dy2)
    angles = (np.arctan2(dy2, dx2) + 2 * np.pi) % (2 * np.pi)
    df_r['angle_bin'] = (angles / (2 * np.pi) * N_ANGLE_BINS).astype(int)

    # 6) Ray‐march
    df_r['visible'] = False
    for _, ray in df_r.groupby('angle_bin'):
        ray_sorted = ray.sort_values('dist2d')
        blocked = False
        for idx, tgt in ray_sorted.iterrows():
            if blocked:
                continue

            if tgt.cell_id == REFERENCE_ID:
                df_r.at[idx, 'visible'] = True
                continue

            if is_line_of_sight_clear(obs, tgt, occupied, step=STEP_SIZE):
                df_r.at[idx, 'visible'] = True
            else:
                df_r.at[idx, 'visible'] = False
                blocked = True

    # 7) Export to CSV
    df_vis = df_r[df_r.visible]
    df_vis.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df_vis)} visible voxels to {OUTPUT_CSV}")

    # 8) Plot
    plot_3d(obs, df_r, df_vis)

if __name__ == "__main__":
    main()