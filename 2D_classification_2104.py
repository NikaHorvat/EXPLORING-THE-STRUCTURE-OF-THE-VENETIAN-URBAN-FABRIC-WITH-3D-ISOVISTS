import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from shapely.geometry import Point
from shapely.vectorized import contains

def main():
    # 1) Load the three layers
    buildings = gpd.read_file("2024_Edifici.geojson")
    streets   = gpd.read_file("2024_Streets.geojson")
    canals    = gpd.read_file("2024_Canals.geojson")

    # 2) Isolate the building with EDIFI_ID = 1459 and make a 200 m buffer
    target = buildings[ buildings["EDIFI_ID"] == 2104 ]

    geom_1459 = target.geometry.iloc[0]
    buffer_200m = geom_1459.buffer(200)

    minx_buf, miny_buf, maxx_buf, maxy_buf = buffer_200m.bounds

    # 3) Precompute unions
    buildings_union = buildings.geometry.unary_union
    streets_union   = streets.geometry.unary_union
    canals_union    = canals.geometry.unary_union

    # 4) Build a 1 m grid
    cell_size = 1

    # Compute how many columns/rows are needed:
    nx = int(np.ceil((maxx_buf - minx_buf) / cell_size))
    ny = int(np.ceil((maxy_buf - miny_buf) / cell_size))

    # Generate the X and Y coordinates of all 1 m‐cell centers
    x_centers = minx_buf + (cell_size / 2.0) + np.arange(nx) * cell_size
    y_centers = miny_buf + (cell_size / 2.0) + np.arange(ny) * cell_size
    X, Y = np.meshgrid(x_centers, y_centers)
    X_flat, Y_flat = X.ravel(), Y.ravel()

    # Filter to only those centers inside the 200 m buffer
    buffer_mask = contains(buffer_200m, X_flat, Y_flat)
    X_buf = X_flat[buffer_mask]
    Y_buf = Y_flat[buffer_mask]

    # 6) Classify each center
    building_mask = contains(buildings_union, X_buf, Y_buf)
    street_mask   = contains(streets_union,   X_buf, Y_buf)
    canal_mask    = contains(canals_union,    X_buf, Y_buf)

    facade_mask  = building_mask & street_mask
    building_only = building_mask
    street_only   = street_mask
    canal_only    = canal_mask

    classification = np.where(
        facade_mask, "facade",
        np.where(building_only, "building",
        np.where(street_only,   "street",
        np.where(canal_only,    "canal", "nothing")))
    )

    # 7) Build a DataFrame of all buffer‐grid centers
    df = pd.DataFrame({
        "cell_id": np.arange(1, len(X_buf) + 1),
        "x_center": X_buf,
        "y_center": Y_buf,
        "classification": classification
    })

    # 8) Keep only the non‐empty cells
    df_nonempty = df[df["classification"] != "nothing"].copy()

    # 9) Build a GeoDataFrame
    gdf_nonempty = gpd.GeoDataFrame(
        df_nonempty,
        geometry=[Point(xy) for xy in zip(df_nonempty["x_center"], df_nonempty["y_center"])],
        crs=buildings.crs  # still EPSG:32633
    )

    # 10) Function to extract a layer’s IDs
    def get_uid(layer, gdf, key):
        overlay = gpd.overlay(gdf, layer[[key, 'geometry']], how='intersection')
        if overlay.empty:
            return pd.DataFrame(columns=['cell_id', key])
        return (
            overlay.groupby('cell_id')[key]
                   .agg(lambda x: ', '.join(map(str, x.unique())))
                   .reset_index()
        )

    b_uids = get_uid(buildings, gdf_nonempty, 'EDIFI_ID')
    s_uids = get_uid(streets,   gdf_nonempty, 'uid')
    c_uids = get_uid(canals,    gdf_nonempty, 'uid')

    # 11) Merge those UIDs back into df_nonempty
    df_nonempty = df_nonempty.merge(b_uids, on='cell_id', how='left')
    df_nonempty = df_nonempty.merge(s_uids.rename(columns={'uid':'uid_s'}), on='cell_id', how='left')
    df_nonempty = df_nonempty.merge(c_uids.rename(columns={'uid':'uid_c'}), on='cell_id', how='left')

    # 12) Assign a single 'uid' per cell based on its classification
    def assign_uid(row):
        cls = row['classification']
        if cls == 'facade':
            b = row.get('EDIFI_ID', '')
            s = row.get('uid_s', '')
            return f"{b},{s}" if b and s else (b or s)
        if cls == 'building':
            return row.get('EDIFI_ID')
        if cls == 'street':
            return row.get('uid_s')
        if cls == 'canal':
            return row.get('uid_c')
        return None

    df_nonempty['uid'] = df_nonempty.apply(assign_uid, axis=1)

    # 13) Save to CSV
    df_nonempty.to_csv('2D_classification_around_2104.csv', index=False)
    print(f"Exported {len(df_nonempty)} rows (non-empty cells) to 2D_classification_around_2104.csv")

    # 14) Visualize the 1 m grid within the 200 m circle
    mapping = {"nothing": 0, "facade": 1, "building": 2, "street": 3, "canal": 4}
    numeric_grid = np.zeros((ny, nx), dtype=int)
    xs_idx = ((X_buf - minx_buf) // cell_size).astype(int)
    ys_idx = ((Y_buf - miny_buf) // cell_size).astype(int)

    for i, cls in enumerate(classification):
        r = ys_idx[i]
        c = xs_idx[i]
        numeric_grid[r, c] = mapping[cls]

    cmap = mcolors.ListedColormap(["white", "orange", "lightgray", "black", "lightblue"])
    norm = mcolors.BoundaryNorm([0,1,2,3,4,5], cmap.N)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title("1 m 2D Classification within 200 m of EDIFI_ID = 2104", fontsize=14)
    ax.imshow(
        numeric_grid,
        cmap=cmap,
        norm=norm,
        origin='lower',
        extent=(minx_buf, maxx_buf, miny_buf, maxy_buf)
    )

    patches = [
        mpatches.Patch(color="white", label="nothing"),
        mpatches.Patch(color="orange", label="facade"),
        mpatches.Patch(color="lightgray", label="building"),
        mpatches.Patch(color="black", label="street"),
        mpatches.Patch(color="lightblue", label="canal")
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()