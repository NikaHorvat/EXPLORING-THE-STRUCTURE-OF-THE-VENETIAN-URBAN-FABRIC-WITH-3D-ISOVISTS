import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from shapely.geometry import Point
from shapely.vectorized import contains

def main():
    # 1) Load GeoJSON files
    buildings = gpd.read_file("2024_Edifici.geojson")
    streets   = gpd.read_file("2024_Streets.geojson")
    canals    = gpd.read_file("2024_Canals.geojson")

    # 2) Union geometries
    buildings_union = buildings.geometry.unary_union
    streets_union   = streets.geometry.unary_union
    canals_union    = canals.geometry.unary_union

    # 3) Compute overall bounding box
    all_union = buildings_union.union(streets_union).union(canals_union)
    minx, miny, maxx, maxy = all_union.bounds

    # 4) Set up grid of cell centers
    cell_size = 7
    nx = int(np.ceil((maxx - minx) / cell_size))
    ny = int(np.ceil((maxy - miny) / cell_size))
    x_centers = minx + cell_size/2 + np.arange(nx) * cell_size
    y_centers = miny + cell_size/2 + np.arange(ny) * cell_size
    X, Y = np.meshgrid(x_centers, y_centers)
    X_flat, Y_flat = X.ravel(), Y.ravel()

    # 5) Classify each center
    building_mask = contains(buildings_union, X_flat, Y_flat)
    street_mask   = contains(streets_union,   X_flat, Y_flat)
    canal_mask    = contains(canals_union,    X_flat, Y_flat)

    facade_mask   = building_mask & street_mask
    building_only = building_mask
    street_only   = street_mask
    canal_only    = canal_mask

    classification = np.where(
        facade_mask, "facade",
        np.where(building_only, "building",
        np.where(street_only,   "street",
        np.where(canal_only,    "canal", "nothing")))
    )
    classification_grid = classification.reshape(ny, nx)

    # 6) Prepare DataFrame of non-empty cells
    df = pd.DataFrame({
        "cell_id": np.arange(1, nx*ny + 1),
        "center": list(zip(X_flat, Y_flat)),
        "classification": classification
    })
    df_nonempty = df[df["classification"] != "nothing"].copy()

    # 7) Build GeoDataFrame for non-empty cells
    gdf_nonempty = gpd.GeoDataFrame(
        df_nonempty,
        geometry=[Point(xy) for xy in df_nonempty["center"]],
        crs=buildings.crs
    )

    # 8) Function to extract UIDs via intersection
    def get_uid(layer, gdf, key):
        overlay = gpd.overlay(gdf, layer[[key, 'geometry']], how='intersection')
        if overlay.empty:
            return pd.DataFrame(columns=['cell_id', key])
        return (
            overlay.groupby('cell_id')[key]
                .agg(lambda x: ', '.join(map(str, x.unique())))
                .reset_index()
        )

    # Pull UIDs per layer
    b_uids = get_uid(buildings, gdf_nonempty, 'EDIFI_ID')
    s_uids = get_uid(streets,   gdf_nonempty, 'uid')
    c_uids = get_uid(canals,    gdf_nonempty, 'uid')

    # 9) Merge UIDs back into df_nonempty
    df_nonempty = df_nonempty.merge(b_uids, on='cell_id', how='left')
    df_nonempty = df_nonempty.merge(s_uids.rename(columns={'uid':'uid_s'}), on='cell_id', how='left')
    df_nonempty = df_nonempty.merge(c_uids.rename(columns={'uid':'uid_c'}), on='cell_id', how='left')

    # 10) Assign a single uid per cell
    def assign_uid(row):
        cls = row['classification']
        if cls == 'facade':
            return row.get('EDIFI_ID', '') + ',' + row.get('uid_s', '')
        if cls == 'building':
            return row.get('EDIFI_ID')
        if cls == 'street':
            return row.get('uid_s')
        if cls == 'canal':
            return row.get('uid_c')
        return None

    df_nonempty['uid'] = df_nonempty.apply(assign_uid, axis=1)

    # 11) Save to CSV
    df_nonempty.to_csv('2D_classification.csv', index=False)
    print(f"Exported {len(df_nonempty)} rows to 2D_classification.csv")

    # 12) Visualization
    mapping = {"nothing": 0, "facade": 1, "building": 2, "street": 3, "canal": 4}
    numeric = np.vectorize(mapping.get)(classification_grid)

    cmap = mcolors.ListedColormap(["white", "orange", "lightgray", "black", "lightblue"])
    norm = mcolors.BoundaryNorm([0,1,2,3,4,5], cmap.N)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_title("2D Classification of Venice", fontsize=14)
    ax.imshow(numeric, cmap=cmap, norm=norm, origin='lower', extent=(minx,maxx,miny,maxy))

    patches = [
        mpatches.Patch(color="white", label="nothing"),
        mpatches.Patch(color="orange", label="facade"),
        mpatches.Patch(color="lightgray", label="building"),
        mpatches.Patch(color="black", label="street"),
        mpatches.Patch(color="lightblue", label="canal")
    ]
    ax.legend(handles=patches, bbox_to_anchor=(1.05,1), loc='upper left')
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()