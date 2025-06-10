import math
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point

def main():
    # 1) Load the 2D classification CSV file
    df_2d = pd.read_csv("2D_classification.csv", low_memory=False)

    # 2) Load GeoJSON files
    gdf_buildings = gpd.read_file("2024_Edifici.geojson")
    gdf_facades   = gpd.read_file("2024_Facades.geojson")

    # 3) Isolate the building with EDIFI_ID = 2104 and create a 200 m buffer around it
    gdf_b2104 = gdf_buildings[gdf_buildings["EDIFI_ID"].astype(str) == "2104"]
    if gdf_b2104.empty:
        raise ValueError("No building with EDIFI_ID = 2104 found in 2024_Edifici.geojson.")
    building_geom = gdf_b2104.unary_union
    buffer_200m   = building_geom.buffer(200.0)

    # 4) Turn the "center" column into Point geometries
    points = []
    for c in df_2d["center"]:
        if isinstance(c, str):
            stripped = c.strip("()")
            x_str, y_str = stripped.split(",")
            x_val, y_val = float(x_str), float(y_str)
        else:
            x_val, y_val = c
        points.append(Point(x_val, y_val))

    df_2d["pt_geom"] = points
    # Create a GeoDataFrame with "pt_geom" as its active geometry
    gdf_points = gpd.GeoDataFrame(df_2d, geometry="pt_geom", crs=gdf_buildings.crs)

    # 5) Filter points within the 200 m buffer
    gdf_200m = gdf_points[gdf_points.within(buffer_200m)].copy()
    if gdf_200m.empty:
        print("No 2D tiles found within 200 m of building 2104.")
        return

    # 6) Identify facades intersecting the 200 m tiles
    gdf_facades_sub = gdf_facades[["FAC_UID", "height", "geometry"]].copy()
    gdf_facades_sub = gdf_facades_sub.set_crs(gdf_200m.crs, allow_override=True)
    
    gdf_left = gdf_200m.set_geometry("pt_geom")
    gdf_left = gdf_left[["cell_id", "pt_geom"]].rename_geometry("geometry")
    
    gdf_tile_facade = gpd.sjoin(
        gdf_left,
        gdf_facades_sub,
        how="inner",
        predicate="intersects"
    )

    # For each tile take the maximum facade height
    if not gdf_tile_facade.empty:
        facade_heights = (
            gdf_tile_facade.groupby("cell_id")["height"]
            .max()
            .reset_index()
            .rename(columns={"height": "facade_height"})
        )
    else:
        facade_heights = pd.DataFrame(columns=["cell_id", "facade_height"])

    # 7) Process building heights
    bldg_heights = gdf_buildings[["EDIFI_ID", "max_height"]].copy()
    bldg_heights["EDIFI_ID"] = bldg_heights["EDIFI_ID"].astype(str)

    df_2d_200m = pd.DataFrame(gdf_200m.drop(columns="pt_geom"))
    df_2d_200m["EDIFI_ID"] = df_2d_200m["EDIFI_ID"].astype(str)

    # 9) Merge building/facade heights
    df_2d_200m = df_2d_200m.merge(
        bldg_heights,
        on="EDIFI_ID",
        how="left"
    )

    df_2d_200m = df_2d_200m.merge(
        facade_heights,
        on="cell_id",
        how="left"
    )

    # 11) Overwrite "classification" and create "use_height"
    def assign_true_class_and_height(r):
        fh = r.get("facade_height", np.nan)
        if not pd.isna(fh):
            return ("facade", fh)
        else:
            return (r["classification"], r.get("max_height", np.nan))

    df_2d_200m[["classification", "use_height"]] = df_2d_200m.apply(
        lambda r: pd.Series(assign_true_class_and_height(r)),
        axis=1
    )

    # 12) Build 3D voxels from the 2D DataFrame
    voxel_3d_rows = []
    for row in df_2d_200m.itertuples(index=False):
        row_dict = row._asdict()
        cls       = row_dict["classification"]
        height    = row_dict["use_height"]

        # Extract x_center, y_center
        if isinstance(row_dict["center"], str):
            cs   = row_dict["center"].strip("()")
            xs, ys = cs.split(",")
            x_ctr, y_ctr = float(xs), float(ys)
        else:
            x_ctr, y_ctr = row_dict["center"]

        # If classification is "building" or "facade" and height is valid, extrude in 1 m layers
        if cls in ["building", "facade"] and not pd.isna(height):
            num_layers = int(math.ceil(height / 1.0))
            for z_index in range(num_layers):
                layer = dict(row_dict)
                layer["z_index"]   = z_index
                layer["z_min"]     = z_index * 1.0
                layer["z_max"]     = (z_index + 1) * 1.0
                z_ctr = (layer["z_min"] + layer["z_max"]) / 2.0
                layer["center_3d"] = (x_ctr, y_ctr, z_ctr)
                voxel_3d_rows.append(layer)
        else:
            layer = dict(row_dict)
            layer["z_index"]   = 0
            layer["z_min"]     = 0.0
            layer["z_max"]     = 0.0
            layer["center_3d"] = (x_ctr, y_ctr, 0.0)
            voxel_3d_rows.append(layer)

    # 13) Build final DataFrame
    df_3d_200m = pd.DataFrame(voxel_3d_rows)
    df_3d_200m.to_csv("3D_classification_around_2104.csv", index=False)

if __name__ == "__main__":
    main()