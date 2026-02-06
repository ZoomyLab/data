#!/usr/bin/env python3
"""
Convert TELEMAC SLF to Gmsh .msh with interpolation.
Usage: python slf2msh_interp.py target_mesh.slf --interpolate source_results.slf
"""

import xarray as xr
import numpy as np
import os
import sys
import argparse
from scipy.interpolate import griddata


# ---------------------------------------------------------------------
# 1. Helper: Write Gmsh NodeData
# ---------------------------------------------------------------------
def write_node_data(f, name, values, timestep=0):
    values = np.asarray(values)
    n = len(values)

    # Handle scalar vs vector shapes
    if values.ndim == 1:
        ncomp = 1
    else:
        ncomp = values.shape[1]

    f.write("$NodeData\n")
    f.write("1\n")  # Num string tags
    f.write(f'"{name}"\n')  # Name
    f.write("1\n")  # Num real tags
    f.write(f"{float(timestep)}\n")
    f.write("3\n")  # Num int tags
    f.write(f"{timestep}\n")  # Time step
    f.write(f"{ncomp}\n")  # Num components
    f.write(f"{n}\n")  # Num nodes

    if ncomp == 1:
        for i, v in enumerate(values, start=1):
            f.write(f"{i} {float(v)}\n")
    else:
        for i, row in enumerate(values, start=1):
            row_floats = " ".join(str(float(x)) for x in row)
            f.write(f"{i} {row_floats}\n")

    f.write("$EndNodeData\n")


# ---------------------------------------------------------------------
# 2. Helper: Write Full MSH
# ---------------------------------------------------------------------
def write_gmsh_with_fields(x, y, tri, fields, outfile, timestep=0):
    with open(outfile, "w") as f:
        # Header
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # Nodes
        f.write(f"$Nodes\n{len(x)}\n")
        for i, (xx, yy) in enumerate(zip(x, y), start=1):
            # Gmsh expects 3D coords (x, y, z). We set z=0.0
            f.write(f"{i} {xx} {yy} 0.0\n")
        f.write("$EndNodes\n")

        # Elements (Triangles)
        f.write(f"$Elements\n{len(tri)}\n")
        for i, (a, b, c) in enumerate(tri, start=1):
            # 2 = Triangle 3-node, 0 tags, node indices (1-based)
            f.write(f"{i} 2 0 {a + 1} {b + 1} {c + 1}\n")
        f.write("$EndElements\n")

        # Fields
        for name, arr in fields.items():
            print(f" -> Writing field: {name}")
            write_node_data(f, name, arr, timestep)

    print(f"\n[Success] Output written to: {outfile}")


# ---------------------------------------------------------------------
# 3. Main Logic
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Map fields from a source SLF to a target SLF and export to Gmsh."
    )
    parser.add_argument("infile", help="Target SLF (Geometry/Fine Mesh)")
    parser.add_argument(
        "-i", "--interpolate", required=True, help="Source SLF (Coarse Results)"
    )
    parser.add_argument(
        "-t",
        "--timestep",
        type=int,
        default=-1,
        help="Timestep to extract (-1 for last step, default)",
    )
    args = parser.parse_args()

    # --- Load Target (Geometry) ---
    print(f"[1/3] Loading Target Mesh: {args.infile}")
    try:
        ds_tgt = xr.open_dataset(args.infile, engine="selafin")
    except Exception as e:
        print(
            f"Error: Could not open {args.infile}. Ensure 'selafin' engine is installed."
        )
        sys.exit(1)

    x_tgt = ds_tgt["x"].values
    y_tgt = ds_tgt["y"].values

    # Handle connectivity (ikle2)
    if "ikle2" in ds_tgt.attrs:
        tri = np.asarray(ds_tgt.attrs["ikle2"], dtype=int) - 1
    else:
        print("Error: Target file missing 'ikle2' connectivity attribute.")
        sys.exit(1)

    # Initialize fields with whatever exists in Target (e.g., Bottom)
    fields = {}
    # Filter out coords/time
    tgt_vars = [v for v in ds_tgt.variables if v not in ("x", "y", "time")]

    # Load target data (usually only Geometry/Bottom)
    for name in tgt_vars:
        val = ds_tgt[name].values
        # If variable has time dimension, take the requested step (or 0)
        if val.ndim > 1:
            idx = args.timestep if args.timestep >= 0 else -1
            fields[name] = val[idx]
        else:
            fields[name] = val

    # --- Load Source (Results) & Interpolate ---
    if args.interpolate:
        print(f"[2/3] Interpolating from: {args.interpolate}")
        ds_src = xr.open_dataset(args.interpolate, engine="selafin")

        x_src = ds_src["x"].values
        y_src = ds_src["y"].values

        # Prepare coordinates for Griddata (N, 2)
        points_src = np.column_stack((x_src, y_src))
        points_tgt = np.column_stack((x_tgt, y_tgt))

        src_vars = [v for v in ds_src.variables if v not in ("x", "y", "time")]

        # Determine source timestep
        t_idx = args.timestep if args.timestep >= 0 else -1
        real_time = ds_src["time"].values[t_idx]
        print(f"      Using Source Time: {real_time} s (Index: {t_idx})")

        for name in src_vars:
            # ONLY interpolate if target doesn't already have it
            # This protects the high-res 'BOTTOM' from being overwritten by low-res 'BOTTOM'
            if name not in fields:
                print(f"      ... Interpolating {name}")
                val_src = ds_src[name].values

                # Extract specific timestep data
                data_src = val_src[t_idx] if val_src.ndim > 1 else val_src

                # 1. Linear Interpolation (Accurate inside hull)
                interp_val = griddata(points_src, data_src, points_tgt, method="linear")

                # 2. Nearest Neighbor (Fill NaNs at boundaries)
                if np.isnan(interp_val).any():
                    mask_nan = np.isnan(interp_val)
                    fill_val = griddata(
                        points_src, data_src, points_tgt[mask_nan], method="nearest"
                    )
                    interp_val[mask_nan] = fill_val

                fields[name] = interp_val
            else:
                print(f"      ... Skipping {name} (exists in target)")

    # --- Write Output ---
    outfile = os.path.splitext(args.infile)[0] + ".msh"
    print(f"[3/3] Exporting to Gmsh...")
    write_gmsh_with_fields(x_tgt, y_tgt, tri, fields, outfile, timestep=0)


if __name__ == "__main__":
    main()
