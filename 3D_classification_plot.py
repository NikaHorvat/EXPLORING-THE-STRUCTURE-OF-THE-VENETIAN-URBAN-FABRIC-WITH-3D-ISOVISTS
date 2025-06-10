import geopandas as gpd
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import box
from matplotlib.lines import Line2D

def main():
    # 1) Load GeoJSON files
    buildings = gpd.read_file("2024_Edifici.geojson")
    streets   = gpd.read_file("2024_Streets.geojson")
    canals    = gpd.read_file("2024_Canals.geojson")
    facades   = gpd.read_file("2024_Facades.geojson")

    # 2) Determine overall bounds
    minx = min(buildings.total_bounds[0], streets.total_bounds[0],
               canals.total_bounds[0], facades.total_bounds[0])
    miny = min(buildings.total_bounds[1], streets.total_bounds[1],
               canals.total_bounds[1], facades.total_bounds[1])
    maxx = max(buildings.total_bounds[2], streets.total_bounds[2],
               canals.total_bounds[2], facades.total_bounds[2])
    maxy = max(buildings.total_bounds[3], streets.total_bounds[3],
               canals.total_bounds[3], facades.total_bounds[3])

    # 3) Create a grid of 7×7 m squares over the city bounds
    grid_cells = []
    x_coords = np.arange(minx, maxx, 7)
    y_coords = np.arange(miny, maxy, 7)
    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x, y, x + 7, y + 7))
    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=buildings.crs)

    # 4) Calculate the center of each grid cell
    grid['center'] = grid.geometry.centroid
    grid['center_x'] = grid['center'].x
    grid['center_y'] = grid['center'].y

    # Helper function for spatial join
    def spatial_join_and_extract(grid_gdf, layer_gdf, uid_field, class_label, height_field=None, default_height=0):
        cols = [uid_field, 'geometry']
        if height_field:
            cols.append(height_field)
        joined = gpd.sjoin(grid_gdf, layer_gdf[cols], how='inner', predicate='intersects')
        df = pd.DataFrame({
            'center_x': joined['center_x'],
            'center_y': joined['center_y'],
            'classification': class_label,
            'uid': joined[uid_field]
        })
        if height_field:
            df['height'] = joined[height_field]
        else:
            df['height'] = default_height
        return df

    # 5) Spatial joins
    df_buildings = spatial_join_and_extract(grid, buildings, 'EDIFI_ID', 'building', height_field='max_height')
    df_facades   = spatial_join_and_extract(grid, facades, 'FAC_UID', 'facade', height_field='height')
    df_streets   = spatial_join_and_extract(grid, streets, 'uid', 'street', default_height=0)
    df_canals    = spatial_join_and_extract(grid, canals, 'uid', 'canal',  default_height=0)

    # 6) Coarse downsampling for canal cells
    def keep_coarse_canal(row):
        if row['classification'] != 'canal':
            return True
        x_int = int(math.floor(row['center_x']))
        y_int = int(math.floor(row['center_y']))
        return (x_int % 10 == 0) and (y_int % 10 == 0)

    # Combine all dataframes
    df_all = pd.concat([df_buildings, df_facades, df_streets, df_canals], ignore_index=True)
    df_all = df_all[df_all.apply(keep_coarse_canal, axis=1)].copy()

    # 7) Generate 3D voxel layers
    voxel_rows = []
    for row in df_all.itertuples(index=False):
        classification = row.classification
        uid = row.uid
        cx, cy = row.center_x, row.center_y
        height = 0 if pd.isna(row.height) else row.height

        if classification == 'facade' and height > 0:
            n_layers = int(math.ceil(height / 7))
            for layer_idx in range(n_layers):
                z_min = layer_idx * 7
                z_max = (layer_idx + 1) * 7
                z_center = (z_min + z_max) / 2.0
                voxel_rows.append({
                    'center_x': cx,
                    'center_y': cy,
                    'center_z': z_center,
                    'classification': classification,
                    'uid': uid,
                    'z_index': layer_idx,
                    'z_min': z_min,
                    'z_max': z_max,
                    'height': height
                })

        elif classification == 'building' and height > 0:
            n_layers = int(math.ceil(height / 7))
            top_layer_idx = n_layers - 1
            z_min = top_layer_idx * 7
            z_max = (top_layer_idx + 1) * 7
            z_center = (z_min + z_max) / 2.0
            voxel_rows.append({
                'center_x': cx,
                'center_y': cy,
                'center_z': z_center,
                'classification': classification,
                'uid': uid,
                'z_index': top_layer_idx,
                'z_min': z_min,
                'z_max': z_max,
                'height': height
            })

        else:
            voxel_rows.append({
                'center_x': cx,
                'center_y': cy,
                'center_z': 0.0,
                'classification': classification,
                'uid': uid,
                'z_index': 0,
                'z_min': 0,
                'z_max': 0,
                'height': height
            })

    if not voxel_rows:
        print("No voxels generated. Check your input data.")
        return

    # Create DataFrame
    df_voxels = pd.DataFrame(voxel_rows)

    # 8) Sort and assign ID
    df_voxels.sort_values(by=['center_y', 'center_x', 'z_index'], inplace=True)
    df_voxels.reset_index(drop=True, inplace=True)
    df_voxels.insert(0, 'cell_id', range(1, len(df_voxels) + 1))

    # 9) Export
    df_voxels.to_csv("3D_classification_plot.csv", index=False)
    print(f"3D voxel table created with {len(df_voxels)} rows.")

    # 10) Plot
    x_vals = df_voxels['center_x'].values
    y_vals = df_voxels['center_y'].values
    z_vals = df_voxels['center_z'].values

    color_map = {
        'building': 'grey',
        'facade':   'orange',
        'canal':    'lightblue',
        'street':   'black'
    }
    colors = [color_map.get(cls, "white") for cls in df_voxels['classification']]

    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(x_vals, y_vals, z_vals,
               c=colors,
               s=5,
               alpha=0.7,
               linewidth=0)

    # axis labels
    ax.set_xlabel("X coordinate (m)", labelpad=10, fontsize=12)
    ax.set_ylabel("Y coordinate (m)", labelpad=10, fontsize=12)
    ax.set_zlabel("Z coordinate (m)", labelpad=10, fontsize=12)

    ax.view_init(elev=25, azim=-60)

    ax.grid(True, linestyle=':', color='gray', alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Building',
               markerfacecolor='grey', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Facade',
               markerfacecolor='orange', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Canal',
               markerfacecolor='lightblue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Street',
               markerfacecolor='black', markersize=8),
    ]
    ax.legend(handles=legend_elements,
              loc='upper left',
              bbox_to_anchor=(0.02, 0.98),
              framealpha=0.8)

    plt.title("3D Voxel Plot of Venice", pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()