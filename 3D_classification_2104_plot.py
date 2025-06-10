import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def main():
    # 1) Input/Output CSV files
    INPUT_CSV = "3D_classification_around_2104.csv"
    OUTPUT_CSV = "3D_classification_ext_around_2104.csv"
    
    # 2) Load the CSV file
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"Loaded {len(df)} 3D voxels from '{INPUT_CSV}'.")
    
    # 3) Compute x_center, y_center, z_center
    if "x_center" not in df.columns or "y_center" not in df.columns:
        # Check if we have x_min, x_max, y_min, y_max
        if "center_3d" in df.columns:
            def parse_xy(c3d: str):
                s = c3d.strip().lstrip("(").rstrip(")")
                parts = [p.strip() for p in s.split(",")]
                return float(parts[0]), float(parts[1])
            xy = df["center_3d"].apply(parse_xy)
            df["x_center"] = xy.apply(lambda t: t[0])
            df["y_center"] = xy.apply(lambda t: t[1])
        elif "center" in df.columns:
            def parse_xy2(c2d: str):
                s = c2d.strip().lstrip("(").rstrip(")")
                parts = [p.strip() for p in s.split(",")]
                return float(parts[0]), float(parts[1])
            xy2 = df["center"].apply(parse_xy2)
            df["x_center"] = xy2.apply(lambda t: t[0])
            df["y_center"] = xy2.apply(lambda t: t[1])
        else:
            raise ValueError("Cannot find X/Y coordinates in the CSV.")
    # Check if x_center, y_center are computed
    if "z_center" not in df.columns:
        if "z_min" in df.columns and "z_max" in df.columns:
            df["z_center"] = (df["z_min"] + df["z_max"]) / 2.0
        elif "center_3d" in df.columns:
            def parse_z(c3d: str):
                s = c3d.strip().lstrip("(").rstrip(")")
                parts = [p.strip() for p in s.split(",")]
                return float(parts[2]) if len(parts) == 3 else 0.0
            df["z_center"] = df["center_3d"].apply(parse_z)
        else:
            raise ValueError("Cannot compute Z coordinate: missing z_min/z_max and center_3d.")
        
    # 4) Identify and remove interior building voxels
    bldg_df = df[df["classification"] == "building"].copy()
    coords = set(zip(bldg_df["x_center"], bldg_df["y_center"], bldg_df["z_center"]))

    # check neighbor existence
    def is_enclosed(x, y, z):
        offsets = [
            (7.0,  0.0,  0.0),
            (-7.0, 0.0,  0.0),
            (0.0,  7.0,  0.0),
            (0.0, -7.0,  0.0),
            (0.0,  0.0,  1.0),
            (0.0,  0.0, -1.0),
        ]
        for dx, dy, dz in offsets:
            neighbor = (x + dx, y + dy, z + dz)
            if neighbor not in coords:
                return False
        return True

    bldg_df["is_interior"] = bldg_df.apply(
        lambda r: is_enclosed(r["x_center"], r["y_center"], r["z_center"]),
        axis=1
    )

    # Build a mask of all rows
    mask_non_building = df["classification"] != "building"
    mask_building_exterior = df.apply(
        lambda r: (r["classification"] == "building") and not bldg_df.loc[
            (bldg_df["x_center"] == r["x_center"]) &
            (bldg_df["y_center"] == r["y_center"]) &
            (bldg_df["z_center"] == r["z_center"]),
            "is_interior"
        ].values[0],
        axis=1
    )
    keep_mask = mask_non_building | mask_building_exterior
    df_filtered = df[keep_mask].copy()

    print(f"Filtered out interior building voxels: {len(df) - len(df_filtered)} removed, {len(df_filtered)} remain.")
    
    # 5) Save the filtered DataFrame to a new CSV
    df_filtered.to_csv(OUTPUT_CSV, index=False)
    print(f"Exported filtered exterior-only voxels to '{OUTPUT_CSV}'.")
    
    # 6) Extract coordinates for plotting
    x_vals = df_filtered["x_center"].values
    y_vals = df_filtered["y_center"].values
    z_vals = df_filtered["z_center"].values
    # 7) Assign colors based on classification
    color_map = {
        "building": "grey",
        "facade":   "grey",
        "canal":    "lightblue",
        "street":   "black"
    }
    colors = [color_map.get(c, "white") for c in df_filtered["classification"].values]
    
    # 8) Make a 3D scatter plot
    fig = plt.figure(figsize=(14, 10))
    ax  = fig.add_subplot(111, projection="3d")

    ax.scatter(
        x_vals,
        y_vals,
        z_vals,
        c=colors,
        s=6,
        alpha=0.7,
        linewidth=0
    )

    ax.set_xlabel("X coordinate (m)", labelpad=10, fontsize=12)
    ax.set_ylabel("Y coordinate (m)", labelpad=10, fontsize=12)
    ax.set_zlabel("Z coordinate (m)", labelpad=10, fontsize=12)
    ax.view_init(elev=25, azim=-60)
    ax.grid(True, linestyle=":", color="gray", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Building exterior",
               markerfacecolor="grey", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Facade",
               markerfacecolor="grey", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Canal",
               markerfacecolor="lightblue", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Street",
               markerfacecolor="black", markersize=8),
    ]
    ax.legend(handles=legend_elements,
              loc="upper left",
              bbox_to_anchor=(0.02, 0.98),
              framealpha=0.8)

    plt.title("3D Voxels around EDIFI_ID = 2104", pad=20, fontsize=14)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()