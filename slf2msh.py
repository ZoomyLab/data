#!/usr/bin/env python3
"""
Convert a TELEMAC SELAFIN (.slf) file into a Gmsh v2.2 .msh mesh file,
including nodal fields stored as $NodeData blocks.

Usage:
    python slf_to_msh.py file.slf [timestep]
"""

import xarray as xr
import numpy as np
import os
import sys


# ---------------------------------------------------------------------
# Helper: write Gmsh $NodeData block
# ---------------------------------------------------------------------
def write_node_data(f, name, values, timestep=0):
    """
    Write a scalar or vector nodal field to Gmsh $NodeData.
    values shape can be:
        (n,)          – scalar
        (n, 2)        – vector of 2 components
        (n, k)        – vector of k components
    """
    values = np.asarray(values)
    n = len(values)

    # components count
    if values.ndim == 1:
        ncomp = 1
    else:
        ncomp = values.shape[1]

    f.write("$NodeData\n")
    f.write("1\n")             # number of string tags
    f.write(f"\"{name}\"\n")   # field name

    f.write("1\n")             # number of real tags
    f.write(f"{float(timestep)}\n")  # time value (or 0)

    f.write("3\n")             # number of integer tags
    f.write(f"{timestep}\n")   # time step index
    f.write(f"{ncomp}\n")      # number of components
    f.write(f"{n}\n")          # number of nodes

    # dump nodal data
    if ncomp == 1:
        # scalar
        for i, v in enumerate(values, start=1):
            f.write(f"{i} {float(v)}\n")
    else:
        # vector
        for i, row in enumerate(values, start=1):
            row_floats = " ".join(str(float(x)) for x in row)
            f.write(f"{i} {row_floats}\n")

    f.write("$EndNodeData\n")


# ---------------------------------------------------------------------
# Main: write mesh + fields
# ---------------------------------------------------------------------
def write_gmsh_with_fields(x, y, tri, fields, outfile, timestep=0):
    """
    fields = dict(name -> array)
        each array either shape (n,) or (n,k)
    """
    with open(outfile, "w") as f:
        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        # ----------------------------------------------------
        # Nodes
        # ----------------------------------------------------
        f.write(f"$Nodes\n{len(x)}\n")
        for i, (xx, yy) in enumerate(zip(x, y), start=1):
            f.write(f"{i} {xx} {yy} 0.0\n")
        f.write("$EndNodes\n")

        # ----------------------------------------------------
        # Triangles
        # ----------------------------------------------------
        f.write(f"$Elements\n{len(tri)}\n")
        for i, (a, b, c) in enumerate(tri, start=1):
            f.write(f"{i} 2 0 {a+1} {b+1} {c+1}\n")
        f.write("$EndElements\n")

        # ----------------------------------------------------
        # Node Data (fields)
        # ----------------------------------------------------
        for name, arr in fields.items():
            write_node_data(f, name, arr, timestep)

    print(f"Wrote Gmsh mesh + fields: {outfile}")


# ---------------------------------------------------------------------
# SLF → Gmsh
# ---------------------------------------------------------------------
def slf_to_msh(infile, timestep=0):
    print(f"[slf_to_msh] Reading SLF: {infile}")
    ds = xr.open_dataset(infile, engine="selafin")

    # Node coordinates
    x = ds["x"].values
    y = ds["y"].values
    n = len(x)

    # Connectivity: ikle2 stored in attributes
    if "ikle2" not in ds.attrs:
        raise KeyError("Missing 'ikle2' in SLF file attributes.")
    tri = np.asarray(ds.attrs["ikle2"], dtype=int) - 1

    # Collect fields (everything except coordinates & time)
    field_names = [v for v in ds.variables if v not in ("x", "y", "time")]

    print(f"[slf_to_msh] Exporting fields: {field_names}")

    fields = {}
    for name in field_names:
        arr = ds[name].values
        if arr.ndim == 1:
            fields[name] = arr.astype(float)
        elif arr.ndim == 2:
            fields[name] = arr[timestep].astype(float)
        elif arr.ndim == 3:
            fields[name] = arr[timestep].astype(float)
        else:
            print(f"WARNING: skipping {name}, unsupported shape {arr.shape}")

    # Output path
    outfile = os.path.splitext(infile)[0] + ".msh"

    # Write mesh + fields
    write_gmsh_with_fields(x, y, tri, fields, outfile, timestep)


# ---------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python slf_to_msh.py file.slf [timestep]")
        sys.exit(1)

    infile = sys.argv[1]
    timestep = int(sys.argv[2]) if len(sys.argv) == 3 else 0
    slf_to_msh(infile, timestep)
