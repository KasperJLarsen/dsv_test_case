import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ==========================================
# 1. DIRECTORY CONFIGURATION
# ==========================================
data_path = Path.cwd() / "pickup_delivery_time_windows"
data_files = [f for f in data_path.rglob("*") if f.is_file() and f.suffix != ".txt"]

# Create an output directory for the saved visualization images
output_dir = Path.cwd() / "visualization_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Found {len(data_files)} problem instances to process.")

# ==========================================
# 2. BATCH LOOPING OVER INSTANCES
# ==========================================
for i in range(len(data_files)):
    input_file = data_files[i]  # Original .pdptw problem instance
    solution_file = data_files[i].with_suffix(".txt")  # HeuriGym solver solution text file

    # Define file path for the output chart image
    output_image_path = output_dir / f"{input_file.stem}_route_map.png"

    # Skip processing if no corresponding solution file exists
    if not solution_file.exists():
        print(f"[{i + 1}/{len(data_files)}] Skipping {input_file.name} (Solution file not found)")
        continue

    print(f"[{i + 1}/{len(data_files)}] Processing {input_file.name}...")

    # Clear previous figures from RAM memory before starting a new plot
    plt.clf()
    plt.close('all')

    coords = {}
    pickup_map = {}
    delivery_map = {}
    depot_id = 1

    # 2.1 Parse the Instance File
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  -> Error reading input file: {e}")
        continue

    reading_coords = False
    reading_pd = False

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        if "NODE_COORD_SECTION" in clean_line:
            reading_coords = True
            reading_pd = False
            continue
        elif "PICKUP_AND_DELIVERY_SECTION" in clean_line:
            reading_coords = False
            reading_pd = True
            continue
        elif "DEPOT_SECTION" in clean_line or "EOF" in clean_line:
            reading_coords = False
            reading_pd = False
            continue

        if reading_coords:
            parts = clean_line.split()
            if len(parts) == 3:
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                coords[node_id] = (x, y)

        elif reading_pd:
            parts = clean_line.replace('"', '').replace("'", "").split()
            if len(parts) >= 7:
                node_id = int(parts[0])
                p_id = int(parts[5])
                d_id = int(parts[6])

                if p_id == 0 and d_id != 0:
                    pickup_map[node_id] = d_id
                elif p_id != 0 and d_id == 0:
                    delivery_map[node_id] = p_id

    if depot_id not in coords:
        print(f"  -> Error: Depot ID {depot_id} missing from coordinate file mapping.")
        continue
    depot_coord = coords[depot_id]

    # 2.2 Parse the Solution File
    routes = []
    instance_name = input_file.stem

    try:
        with open(solution_file, 'r') as f:
            sol_lines = f.readlines()
    except Exception as e:
        print(f"  -> Error reading solution file: {e}")
        continue

    for idx, line in enumerate(sol_lines):
        clean_line = line.strip()
        if not clean_line:
            continue
        if idx == 0 and not clean_line.replace(" ", "").isdigit():
            instance_name = clean_line
            continue

        nodes = [int(n) for n in clean_line.split() if n.replace('-', '').isdigit()]
        if nodes:
            routes.append(nodes)

    # ==========================================
    # 3. GENERATING AND SAVING THE PLOT
    # ==========================================
    plt.figure(figsize=(14, 10))
    cmap = plt.cm.get_cmap("tab20", max(len(routes), 1))

    # 3.1 Plot routes and stops
    for r_idx, route in enumerate(routes):
        route_color = cmap(r_idx)
        r_coords = [coords.get(node, depot_coord) for node in route]
        x_val, y_val = zip(*r_coords)

        plt.plot(x_val, y_val, color=route_color, linestyle="-", linewidth=2, alpha=0.8)

        for node in route:
            if node == depot_id or node not in coords:
                continue

            nx, ny = coords[node]
            if node in pickup_map:
                marker = "^"
                color = "green"
            elif node in delivery_map:
                marker = "v"
                color = "red"
            else:
                marker = "o"
                color = "gray"

            plt.scatter(nx, ny, color=color, marker=marker, s=80, edgecolors='k', zorder=3)
            plt.text(nx + 0.3, ny + 0.3, str(node), fontsize=9, weight='bold', zorder=4)

    # 3.2 Draw demand pairing background lines
    for p_node, d_node in pickup_map.items():
        if p_node in coords and d_node in coords:
            px, py = coords[p_node]
            dx, dy = coords[d_node]
            plt.plot([px, dx], [py, dy], color="black", linestyle=":", alpha=0.15, zorder=1)

    # 3.3 Plot Depot
    plt.scatter(depot_coord[0], depot_coord[1], color="blue", marker="s", s=200,
                edgecolors="black", label="Depot (Node 1)", zorder=5)

    # 3.4 Styling and Legend Layout
    plt.title(f"HeuriGym Solver Solution Map: {instance_name}", fontsize=16, weight="bold", pad=15)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=12, markeredgecolor='k',
               label='Depot'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, markeredgecolor='k',
               label='Pickup Location'),
        Line2D([0], [0], marker='v', color='w', markerfacecolor='red', markersize=10, markeredgecolor='k',
               label='Delivery Location'),
        Line2D([0], [0], linestyle=':', color='black', alpha=0.4, label='Pickup $\\rightarrow$ Delivery Job Link')
    ]

    for r_idx in range(len(routes)):
        legend_elements.append(Line2D([0], [0], color=cmap(r_idx), lw=2, label=f"Route {r_idx + 1}"))

    plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.tight_layout()

    # Save visualization PNG directly to disk without UI blocking stalls
    plt.savefig(output_image_path, dpi=150, bbox_inches="tight")
    print(f"  -> Saved chart visualization to: {output_image_path.name}")

print(
    "\nAll available instance visualizations successfully generated and saved inside the 'visualization_outputs/' directory!")

aq = 42