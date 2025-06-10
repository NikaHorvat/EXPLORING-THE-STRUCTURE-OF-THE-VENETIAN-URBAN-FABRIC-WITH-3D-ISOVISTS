import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.vectorized import contains as shp_contains

def main():
    # Parameters
    CENTER_EDIFI_ID = 2104
    BUFFER_RADIUS = 200.0
    VOXEL_SIZE = 1.0
    OUTPUT_3D_CSV = "3D_buildings_around_2104.csv"

    # 1) Load the GeoJSON file
    gdf_buildings = gpd.read_file("2024_Edifici.geojson")
    gdf_buildings["EDIFI_ID"] = gdf_buildings["EDIFI_ID"].astype(str)

    if "max_height" not in gdf_buildings.columns:
        raise RuntimeError("Your GeoJSON must contain a column named 'max_height'.")

    # 2) Find the building with EDIFI_ID = 2104
    center_bldg = gdf_buildings[gdf_buildings["EDIFI_ID"] == str(CENTER_EDIFI_ID)]
    if center_bldg.empty:
        raise RuntimeError(f"No building found with EDIFI_ID = {CENTER_EDIFI_ID}.")

    center_geom = center_bldg.geometry.unary_union
    buffer_200m = center_geom.buffer(BUFFER_RADIUS)

    # 3) Select buildings within 200 m of the center building
    mask_in_buffer = gdf_buildings.geometry.intersects(buffer_200m)
    gdf_nearby = gdf_buildings[mask_in_buffer].copy()

    # 4) Voxelize the buildings
    voxels = []

    for row in gdf_nearby.itertuples(index=False):
        b_id      = row.EDIFI_ID
        b_height  = row.max_height
        footprint = row.geometry

        if pd.isna(b_height) or b_height <= 0:
            continue

        n_layers = int(math.ceil(b_height / VOXEL_SIZE))

        # Build a 1 m grid of candidate centers
        minx, miny, maxx, maxy = footprint.bounds

        x_start = minx + (VOXEL_SIZE / 2.0)
        y_start = miny + (VOXEL_SIZE / 2.0)
        x_centers = np.arange(x_start, maxx, VOXEL_SIZE)
        y_centers = np.arange(y_start, maxy, VOXEL_SIZE)

        if (len(x_centers) == 0) or (len(y_centers) == 0):
            cx, cy = footprint.centroid.x, footprint.centroid.y
            candidate_pts = np.array([[cx, cy]])
        else:
            Xc, Yc = np.meshgrid(x_centers, y_centers)
            candidate_pts = np.vstack([Xc.ravel(), Yc.ravel()]).T

        try:
            mask_inside = shp_contains(footprint, candidate_pts[:,0], candidate_pts[:,1])
        except ImportError:
            mask_inside = np.array([footprint.contains(Point(x,y)) for x,y in candidate_pts])

        inside_centers = candidate_pts[mask_inside]

        if inside_centers.shape[0] == 0:
            cx, cy = footprint.centroid.x, footprint.centroid.y
            inside_centers = np.array([[cx, cy]])

        for (x_ctr, y_ctr) in inside_centers:
            for z_idx in range(n_layers):
                z_min = z_idx * VOXEL_SIZE
                z_max = (z_idx + 1) * VOXEL_SIZE
                z_ctr = (z_min + z_max) / 2.0
                voxels.append({
                    "EDIFI_ID":  b_id,
                    "x_center":  x_ctr,
                    "y_center":  y_ctr,
                    "z_index":   z_idx,
                    "z_min":     z_min,
                    "z_max":     z_max,
                    "z_center":  z_ctr
                })

    # 5) Export to a CSV file
    df_vox = pd.DataFrame(voxels)
    df_vox.to_csv(OUTPUT_3D_CSV, index=False)
    print(f"Exported {len(df_vox)} total building voxels to '{OUTPUT_3D_CSV}'.")

if __name__ == "__main__":
    main()