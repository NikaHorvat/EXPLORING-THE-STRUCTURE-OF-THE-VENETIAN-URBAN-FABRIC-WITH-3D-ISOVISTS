import math
import pandas as pd
import geopandas as gpd
import numpy as np

def main():
    # 1) Load the final 2D voxel classification output
    df_2d = pd.read_csv("2D_classification.csv", low_memory=False)

    # 2) Load the buildings layer
    gdf_buildings = gpd.read_file("2024_Edifici.geojson") 

    # Keep only the columns we need
    buildings_height = gdf_buildings[["EDIFI_ID", "max_height"]].copy()

    df_2d["EDIFI_ID"] = df_2d["EDIFI_ID"].astype(str)
    buildings_height["EDIFI_ID"] = buildings_height["EDIFI_ID"].astype(str)

    # 3) Merge the 2D classification with the building height data
    df_2d = df_2d.merge(
        buildings_height,
        left_on="EDIFI_ID",
        right_on="EDIFI_ID",
        how="left"
    )

    # 4) Create a new DataFrame for 3D voxels
    voxel_3d_rows = []

    # Iterate over each row in the 2D DataFrame and create 3D voxels 
    for row in df_2d.itertuples(index=False):
        row_dict = row._asdict()
        
        # Extract the classification and building height
        classification = row_dict["classification"]
        building_height = row_dict.get("max_height", np.nan)

        # Extract the 2D center coordinates 
        if isinstance(row_dict["center"], str):
            center_str = row_dict["center"].strip("()")
            x_str, y_str = center_str.split(",")
            x_center, y_center = float(x_str), float(y_str)
        else:
            x_center, y_center = row_dict["center"]

        if classification in ["building", "facade"] and not pd.isna(building_height):
            num_layers = int(math.ceil(building_height / 10.0))
            
            for z_index in range(num_layers):
                layer_row = dict(row_dict)
                layer_row["z_index"] = z_index
                layer_row["z_min"] = z_index * 10
                layer_row["z_max"] = (z_index + 1) * 10

                z_center = (layer_row["z_min"] + layer_row["z_max"]) / 2.0

                layer_row["center_3d"] = (x_center, y_center, z_center)

                voxel_3d_rows.append(layer_row)
        else:
            row_dict["z_index"] = 0
            row_dict["z_min"] = 0
            row_dict["z_max"] = 0

            z_center = 0.0
            row_dict["center_3d"] = (x_center, y_center, z_center)

            voxel_3d_rows.append(row_dict)

    # 5) Build a new DataFrame with 3D voxel info
    df_3d = pd.DataFrame(voxel_3d_rows)

    # 6) Save the 3D voxels to a new CSV
    df_3d.to_csv("3D_classification.csv", index=False)
    print(f"3D voxel table created with {len(df_3d)} rows.")

if __name__ == "__main__":
    main()