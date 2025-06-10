import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from joblib import Parallel, delayed
import plotly.graph_objects as go

# parameters
GRID_SIZE     = 40.0
VIEW_RADIUS   = 50.0
STEP_SIZE     = 1.0
VOXEL_SIZE    = 7.0
N_ANGLE_BINS  = 360
OUTPUT_CSV    = "Venice_visibility.csv"

def voxel_coord(x, y, z):
    return (
        int(math.floor(x / VOXEL_SIZE)),
        int(math.floor(y / VOXEL_SIZE)),
        int(round(z / VOXEL_SIZE)),
    )

def is_line_of_sight_clear(obs_xyz, tgt_xyz, occupied_set, step=STEP_SIZE):
    ox, oy, oz = obs_xyz
    tx, ty, tz = tgt_xyz
    dx, dy, dz = tx-ox, ty-oy, tz-oz
    dist = math.hypot(math.hypot(dx, dy), dz)
    if dist < 1e-6:
        return False
    ux, uy, uz = dx/dist, dy/dist, dz/dist
    target_voxel = voxel_coord(tx, ty, tz)
    offset = 0.01
    steps = int(dist // step)
    for i in range(steps+1):
        d = min(offset + i*step, dist)
        x, y, z = ox + ux*d, oy + uy*d, oz + uz*d
        if d >= dist - 1e-6:
            return True
        v = voxel_coord(x, y, z)
        if v == target_voxel:
            return True
        if v in occupied_set:
            return False
    return True

# Set of occupied voxel coordinates from all building voxels
def process_observer(idx, coords, classification, occupied_buildings_only, view_radius=VIEW_RADIUS):
    obs_xyz = coords[idx]
    dx = coords[:,0] - obs_xyz[0]
    dy = coords[:,1] - obs_xyz[1]
    dist2d = np.hypot(dx, dy)
    within = dist2d <= view_radius
    ids = np.nonzero(within)[0]

    occ_ids = ids[occupied_buildings_only[ids]]
    occ_set = {voxel_coord(*coords[i]) for i in occ_ids}
    angles = (np.arctan2(dy[ids], dx[ids]) + 2*np.pi) % (2*np.pi)
    bins   = np.floor(angles / (2*np.pi) * N_ANGLE_BINS).astype(int)
    vis_local = np.zeros(len(coords), dtype=int)
    for b in range(N_ANGLE_BINS):
        ray = ids[bins == b]
        if ray.size == 0:
            continue
        order = ray[np.argsort(dist2d[ray])]
        for j in order:
            if is_line_of_sight_clear(obs_xyz, coords[j], occ_set):
                if classification[j] == 'facade':
                    vis_local[j] = 1
                    break
                if classification[j] == 'building':
                    break
            else:
                break
    return vis_local


def make_cube_mesh(x, y, z, size=VOXEL_SIZE):
    hs = size/2.0
    verts = np.array([
        [x-hs, y-hs, z-hs], [x+hs, y-hs, z-hs],
        [x+hs, y+hs, z-hs], [x-hs, y+hs, z-hs],
        [x-hs, y-hs, z+hs], [x+hs, y-hs, z+hs],
        [x+hs, y+hs, z+hs], [x-hs, y+hs, z+hs]
    ])
    faces = [(0,1,2),(0,2,3), (4,5,6),(4,6,7),
             (0,1,5),(0,5,4), (2,3,7),(2,7,6),
             (1,2,6),(1,6,5), (4,7,3),(4,3,0)]
    return verts, faces


def main():
    # 1) Load data
    df = pd.read_csv("3D_classification_plot.csv")
    coords = df[['center_x','center_y','center_z']].to_numpy()
    classification = df['classification'].to_numpy()

    # 2) Masks
    building_mask = classification == 'building'
    street_mask   = classification == 'street'
    facade_mask   = classification == 'facade'

    xs, ys = coords[:,0], coords[:,1]
    minx, maxx = xs.min(), xs.max()
    miny, maxy = ys.min(), ys.max()
    gx = np.arange(minx, maxx+GRID_SIZE, GRID_SIZE)
    gy = np.arange(miny, maxy+GRID_SIZE, GRID_SIZE)

    corners = []
    for i in range(len(gx)-1):
        for j in range(len(gy)-1):
            if (i+j) % 2 == 0:
                corners.append((gx[i],   gy[j]))
            else:
                corners.append((gx[i+1], gy[j+1]))
    corners = np.array(corners)

    # 3) Snap corners to nearest street voxel
    street_idx = np.nonzero(street_mask)[0]
    street_pts = coords[street_idx,:2]
    tree_st = cKDTree(street_pts)

    _, nn = tree_st.query(corners)
    obs_indices = np.unique(street_idx[nn])
    print(f"Observers (street voxels): {len(obs_indices)}")

    # Print counts
    print(f"Total facade voxels: {np.sum(facade_mask)}")
    print(f"Total building voxels: {np.sum(building_mask)}")
    print(f"Total street voxels: {np.sum(street_mask)}")

    # 4) Parallel visibility
    vis = Parallel(n_jobs=-1)(
        delayed(process_observer)(i, coords, classification, building_mask)
        for i in obs_indices
    )
    total_counts = np.sum(vis, axis=0)
    df['vis_count'] = total_counts
    df['vis_fraction'] = df['vis_count'] / len(obs_indices)

    fac = df[facade_mask].copy()
    print(f"Facades seen at least once: {np.sum(fac.vis_count>0)} / {len(fac)}")
    print(f"Max visibility count: {fac.vis_count.max()}")
    fac.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported {len(fac)} facade records to {OUTPUT_CSV}")

    # 5) Plot
    all_verts = []
    I = []; J = []; K = []; intensity = []
    for cnt, (x,y,z) in zip(fac.vis_count, fac[['center_x','center_y','center_z']].values):
        verts, faces = make_cube_mesh(x,y,z)
        base = len(all_verts)
        all_verts.extend(verts)
        for a,b,c in faces:
            I.append(base+a); J.append(base+b); K.append(base+c)
            intensity.append(cnt)
    all_verts = np.array(all_verts)

    fig = go.Figure(
        go.Mesh3d(
            x=all_verts[:,0],
            y=all_verts[:,1],
            z=all_verts[:,2],
            i=I, j=J, k=K,
            intensity=intensity,
            colorscale='Inferno',
            cmin=0,
            cmax=5,
            colorbar=dict(title='Visible by # obs'),
            opacity=0.8,
        )
    )

    fig.update_layout(
        title="Venice Facade Visibility (GRID observers)",
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)', zaxis_title='Z (m)', aspectmode='data'),
        width=900, height=700
    )
    fig.show()

if __name__ == '__main__':
    main()