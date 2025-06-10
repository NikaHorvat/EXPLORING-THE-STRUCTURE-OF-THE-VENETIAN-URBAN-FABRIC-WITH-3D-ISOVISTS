import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def main():
    # 1) Load GeoJSON files
    streets = gpd.read_file("2024_Streets.geojson")
    canals = gpd.read_file("2024_Canals.geojson")
    buildings = gpd.read_file("2024_Edifici.geojson")
    facades = gpd.read_file("2024_Facades.geojson")

    # 2) Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))

    # 3) Plot each layer
    buildings.plot(ax=ax, color="lightgray", edgecolor="gray")
    facades.plot(ax=ax, color="none", edgecolor="orange", linewidth=1)
    streets.plot(ax=ax, color="black", linewidth=1)
    canals.plot(ax=ax, color="lightblue", linewidth=1)

    # 4) Label axes
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # 5) Ensure aspect ratio is equal
    ax.set_aspect('equal', adjustable='box')

    # 6) Set axis limits to the combined bounds of all layers
    bnds = [buildings.total_bounds, facades.total_bounds,
            streets.total_bounds, canals.total_bounds]
    minx = min(b[0] for b in bnds)
    miny = min(b[1] for b in bnds)
    maxx = max(b[2] for b in bnds)
    maxy = max(b[3] for b in bnds)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    buildings_handle = mlines.Line2D([], [], color='gray', marker='s', linestyle='None', markersize=10, label='Buildings')
    facades_handle   = mlines.Line2D([], [], color='orange', marker='s', linestyle='None', markersize=10, label='Facades')
    streets_handle   = mlines.Line2D([], [], color='black', linewidth=2, label='Streets')
    canals_handle    = mlines.Line2D([], [], color='lightblue', linewidth=2, label='Canals')

    ax.legend(handles=[buildings_handle, facades_handle, streets_handle, canals_handle], loc='upper right')
    ax.set_title("Map of Venice", fontsize=15)

    plt.show()

if __name__ == "__main__":
    main()