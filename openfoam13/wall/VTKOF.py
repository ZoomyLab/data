import pyvista as pv
import xml.etree.ElementTree as ET
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


def _pv_to_np(self, field_name, normal=None, origin=(0, 0, 0), use_point_data=False):
    """
    Extracts geometric and field data into NumPy arrays.

    Returns:
        1D: (sweep_pos [N], values [N], lengths [N]) -> All 1D NumPy arrays.
        2D: (centers [List of [N,2]], values [List of [N]], lengths [List of [N]])
            -> Each list contains one NumPy array per sweep station.
    """
    if self.n_cells == 0:
        raise ValueError("The mesh has 0 cells.")

    # 1. Automatic Interpolation (CTP / PTC)
    target_mesh = self
    if use_point_data:
        if field_name not in self.point_data:
            if field_name in self.cell_data:
                target_mesh = self.cell_data_to_point_data(pass_cell_data=True)
            else:
                raise KeyError(f"Field '{field_name}' not found in points or cells.")
    else:
        if field_name not in self.cell_data:
            if field_name in self.point_data:
                target_mesh = self.point_data_to_cell_data(pass_point_data=True)
            else:
                raise KeyError(f"Field '{field_name}' not found in cells or points.")

    data_dict = target_mesh.point_data if use_point_data else target_mesh.cell_data
    max_dim = max(
        [target_mesh.get_cell(i).dimension for i in range(min(10, target_mesh.n_cells))]
    )

    # ==========================================
    # 1D Object Logic (The Base Case)
    # ==========================================
    if max_dim == 1:
        pos = (
            target_mesh.points if use_point_data else target_mesh.cell_centers().points
        )
        vals = data_dict[field_name]

        # Determine sweep coordinate
        if normal is not None:
            if isinstance(normal, str):
                axis_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
                n_vec = np.array(axis_map[normal.lower()])
            else:
                n_vec = np.array(normal) / np.linalg.norm(normal)
            sweep_coords = np.dot(pos - np.array(origin), n_vec)
        else:
            dom_axis = np.argmax(np.ptp(pos, axis=0))
            sweep_coords = pos[:, dom_axis]

        sort_idx = np.argsort(sweep_coords)
        sorted_sweep = sweep_coords[sort_idx]
        sorted_vals = vals[sort_idx]

        # Arc-length / Integration weights
        if len(pos) > 1:
            steps = np.linalg.norm(np.diff(pos[sort_idx], axis=0), axis=1)
            arc_dist = np.concatenate(([0], np.cumsum(steps)))
            if use_point_data:
                lens = np.zeros_like(arc_dist)
                lens[0], lens[-1] = steps[0] / 2, steps[-1] / 2
                if len(arc_dist) > 2:
                    lens[1:-1] = (steps[:-1] + steps[1:]) / 2
            else:
                lens = target_mesh.compute_cell_sizes(length=True).cell_data["Length"][
                    sort_idx
                ]
        else:
            arc_dist, lens = np.array([0.0]), np.array([0.0])

        # Logic for returning coordinates
        if normal is None:
            return sorted_sweep, sorted_vals, lens
        else:
            # When recursing from 2D, return the [Sweep, Arc] pair
            return np.column_stack((sorted_sweep, arc_dist)), sorted_vals, lens

    # ==========================================
    # 2D Object Logic (The Sweep Case)
    # ==========================================
    elif max_dim == 2:
        if normal is None:
            raise ValueError("Normal required for 2D sweep.")

        if isinstance(normal, str):
            axis_map = {
                "x": [1.0, 0.0, 0.0],
                "y": [0.0, 1.0, 0.0],
                "z": [0.0, 0.0, 1.0],
            }
            n_vec = np.array(axis_map[normal.lower()])
        else:
            n_vec = np.array(normal) / np.linalg.norm(normal)

        origin_pt = np.array(origin)
        # Find cells along the line
        cell_ids = target_mesh.find_cells_along_line(
            origin_pt - n_vec * 1e6, origin_pt + n_vec * 1e6
        )

        st_pts = target_mesh.cell_centers().points[cell_ids]
        st_projs = np.dot(st_pts - origin_pt, n_vec)
        sort_idx = np.argsort(st_projs)

        c_list, v_list, l_list = [], [], []
        for station in st_pts[sort_idx]:
            slc = target_mesh.slice(normal=n_vec, origin=station)
            if slc.n_cells > 0:
                c, v, l = slc.to_np(
                    field_name,
                    normal=normal,
                    origin=origin,
                    use_point_data=use_point_data,
                )
                c_list.append(c)
                v_list.append(v)
                l_list.append(l)

        # RETURN PURE LISTS (No NaN padding)
        return c_list, v_list, l_list

    else:
        raise ValueError(f".to_np() supports 1D/2D, found {max_dim}.")


def _pv_integrate_mesh(self, normal, origin=(0, 0, 0)):
    """
    Integrates a 2D mesh along a sweep line.
    Powered natively by NumPy arrays extracted via .to_np() for unbreakable stability.
    """
    if self.n_cells == 0:
        raise ValueError("Cannot integrate an empty mesh.")

    # Parse normal for 3D point reconstruction later
    if isinstance(normal, str):
        axis_map = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
        norm_vec = np.array(axis_map[normal.lower()])
    else:
        norm_vec = np.array(normal, dtype=float)
        norm_vec /= np.linalg.norm(norm_vec)

    origin_pt = np.array(origin, dtype=float)

    fields_to_integrate = list(self.cell_data.keys())
    integrated_values = {}
    valid_sweep_projs = None

    # Compute integrals for every field using to_np
    for i, field in enumerate(fields_to_integrate):
        try:
            # 🚀 Extract all local coordinates, values, and lengths cleanly
            centers, values, lengths = self.to_np(field, normal=normal, origin=origin)
        except ValueError:
            continue  # Skip fields that fail slicing

        # Extract the sweep coordinates just once to rebuild the 3D line
        if valid_sweep_projs is None:
            # centers[j][0, 0] accesses the sweep_dist for the j-th slice
            valid_sweep_projs = [c[0, 0] for c in centers]

        field_integrals = []
        # Pure NumPy Riemann Sums!
        for v, dx in zip(values, lengths):
            if v.ndim > 1:  # Handle 3D vector fields cleanly
                field_integrals.append(np.sum(v * dx[:, np.newaxis], axis=0))
            else:  # Handle 1D scalar fields
                field_integrals.append(np.sum(v * dx))

        integrated_values[field] = np.stack(field_integrals)

    if valid_sweep_projs is None or len(valid_sweep_projs) == 0:
        raise RuntimeError("Integration failed. No valid slices were generated.")

    # Reconstruct the 3D points exactly along the sweep line
    valid_points = np.array([origin_pt + proj * norm_vec for proj in valid_sweep_projs])
    n_points = len(valid_points)

    # Build connectivity and the final 1D PyVista Mesh
    lines = np.empty((n_points - 1, 3), dtype=int)
    lines[:, 0] = 2
    lines[:, 1] = np.arange(n_points - 1)
    lines[:, 2] = np.arange(1, n_points)

    line_mesh = pv.PolyData(valid_points, lines=lines.flatten())

    for field, vals in integrated_values.items():
        line_mesh.point_data[field] = vals

    # Clear scalar flag to protect future filters
    line_mesh.point_data.active_scalars_name = None
    return line_mesh


# Attach it natively to PyVista alongside your other custom methods
pv.DataSet.to_np = _pv_to_np
pv.DataSet.integrate = _pv_integrate_mesh


class VTKOF:
    def __init__(self, folder_path, extension=".vtk"):
        self.folder_path = folder_path
        self.extension = extension
        self.file_paths = self._get_sorted_files()
        self.num_timesteps = len(self.file_paths)

        if self.num_timesteps == 0:
            raise FileNotFoundError(
                f"No files ending in '{extension}' found in {folder_path}."
            )

        # Load the exact physical times from the OpenFOAM folders
        self.of_times = self._get_openfoam_times()

        print(f"Initialized OFVTK: Found {self.num_timesteps} files.")
        if len(self.of_times) != self.num_timesteps:
            print(
                f"Warning: Found {len(self.of_times)} OpenFOAM time folders but {self.num_timesteps} VTK files. Time mapping may fallback to filenames."
            )

    def _get_openfoam_times(self):
        """
        Scans the parent directory (OpenFOAM root) for time directories
        (e.g., '0', '0.1', '36.2') and sorts them numerically.
        """
        # The parent of the VTK folder is usually the OpenFOAM root directory
        of_root = os.path.dirname(os.path.abspath(self.folder_path))

        time_folders = []
        import re

        for item in os.listdir(of_root):
            item_path = os.path.join(of_root, item)
            # Check if it's a directory and its name is strictly a number
            if os.path.isdir(item_path) and re.match(r"^[-+]?(?:\d*\.\d+|\d+)$", item):
                time_folders.append(float(item))

        # Sort them numerically so they perfectly align with the VTK files
        time_folders.sort()
        return time_folders

    def _natural_sort_key(self, s):
        """Helper to sort files numerically rather than purely alphabetically."""
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    def _get_sorted_files(self):
        """Collects and sorts the files."""
        search_pattern = os.path.join(self.folder_path, f"*{self.extension}")
        files = glob.glob(search_pattern)
        return sorted(files, key=self._natural_sort_key)

    def _vector_field_to_scalar_field(self, pv_obj, field_name):
        """
        Splits a 3D vector field into three scalar fields (_x, _y, _z)
        and appends them directly to the PyVista object's cell_data.

        Args:
            pv_obj (pyvista.DataSet): The loaded PyVista mesh.
            field_name (str): The name of the vector field (e.g., 'U').
        """
        vector_data = pv_obj.cell_data[field_name]

        pv_obj.cell_data[f"{field_name}_x"] = vector_data[:, 0]
        pv_obj.cell_data[f"{field_name}_y"] = vector_data[:, 1]
        pv_obj.cell_data[f"{field_name}_z"] = vector_data[:, 2]

        vector_data = pv_obj.point_data[field_name]

        pv_obj.point_data[f"{field_name}_x"] = vector_data[:, 0]
        pv_obj.point_data[f"{field_name}_y"] = vector_data[:, 1]
        pv_obj.point_data[f"{field_name}_z"] = vector_data[:, 2]

    def size(self):
        return self.num_timesteps

    def get_time(self, index):
        """
        Returns the physical simulation time for the given timestep index.
        """
        if index < -self.num_timesteps or index >= self.num_timesteps:
            raise IndexError(
                f"Index {index} out of bounds for {self.num_timesteps} timesteps."
            )

        # Handle negative indexing for array lookup
        actual_index = index if index >= 0 else self.num_timesteps + index

        # 1. Primary Method: 1-to-1 mapping with the OpenFOAM root time directories
        if len(self.of_times) == self.num_timesteps:
            return self.of_times[actual_index]

        # 2. Fallback: If folder counts mismatch, fallback to regexing the filename
        file_path = self.file_paths[index]
        basename = os.path.basename(file_path)
        import re

        matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", basename)

        if matches:
            return float(matches[-1])

        raise ValueError(f"Could not determine time for {basename}")

    def get_time_step(self, index):
        """
        Loads the PyVista object for the given timestep index (supports negative indexing)
        and automatically unpacks all vector fields into separate scalar components.
        """
        if index < -self.num_timesteps or index >= self.num_timesteps:
            raise IndexError(
                f"Index {index} out of bounds for {self.num_timesteps} timesteps."
            )

        file_to_load = self.file_paths[index]
        mesh = pv.read(file_to_load)

        # Unpack vector fields into scalars
        # We use list() to avoid modifying the dictionary while iterating over it
        for field_name in list(mesh.cell_data.keys()):
            arr = mesh.cell_data[field_name]

            # Check if the field is a 3D vector (2D array with 3 columns)
            if len(arr.shape) == 2 and arr.shape[1] == 3:
                self._vector_field_to_scalar_field(mesh, field_name)

        return mesh


def get_available_fields(pv_obj):
    """
    Returns a list of available cell data field names for a given PyVista object.

    Args:
        pv_obj (pyvista.DataSet): The PyVista object (e.g., a loaded mesh or a slice).

    Returns:
        list: A list of strings representing the field names.
    """
    return list((list(pv_obj.cell_data.keys()), list(pv_obj.point_data.keys())))


def plot(ax, pv_obj, field_name):
    """
    Plots a 1D or 2D PyVista object using Matplotlib based on cell values.
    Automatically detects if the object is 1D (lines) or 2D (polygons).

    Args:
        pv_obj (pyvista.DataSet): The PyVista object (e.g., from a slice or line sample).
        field_name (str): The name of the cell data field to plot.
        ax (matplotlib.axes.Axes, optional): An existing Matplotlib axis.
    """
    if field_name not in pv_obj.cell_data:
        raise ValueError(
            f"Field '{field_name}' not found in the object's cell_data. "
            f"Available fields: {list(pv_obj.cell_data.keys())}"
        )

    # 1. Determine dimensionality by checking the first few cells
    # (VTK cell dimensions: 1 for lines, 2 for faces/polys)
    sample_size = min(10, pv_obj.n_cells)
    cell_dims = [pv_obj.get_cell(i).dimension for i in range(sample_size)]
    max_dim = max(cell_dims)

    cell_values = pv_obj.cell_data[field_name]

    # ==========================================
    # Handle 1D Object (Line Plot)
    # ==========================================
    if max_dim == 1:
        # Get cell centers
        centers = pv_obj.cell_centers().points

        # To plot correctly, we need a 1D coordinate. We'll use the principal axis
        # (the axis with the largest spread) to sort the points so the line plots linearly.
        spreads = np.ptp(centers, axis=0)
        dominant_axis = np.argmax(spreads)

        # Sort by the dominant spatial coordinate
        sort_indices = np.argsort(centers[:, dominant_axis])

        # Calculate distance along the line (arc length from the first point)
        sorted_centers = centers[sort_indices]
        distances = np.linalg.norm(sorted_centers - sorted_centers[0], axis=1)
        sorted_values = cell_values[sort_indices]

        ax.plot(distances, sorted_values, marker="o", linestyle="-", markersize=4)
        ax.set_xlabel("Distance along line")
        ax.set_ylabel(field_name)
        ax.set_title(f"1D Plot: {field_name}")
        ax.grid(True)

    # ==========================================
    # Handle 2D Object (Planar Mesh Plot)
    # ==========================================
    elif max_dim == 2:
        points = pv_obj.points

        # Since it's a 3D object lying on a 2D plane, find the two axes with the largest spread
        # to project it cleanly onto a 2D Matplotlib axis.
        spreads = np.ptp(points, axis=0)
        planar_axes = np.argsort(spreads)[1:]  # The two axes with the largest variance

        pts_2d = points[:, planar_axes]

        # Extract the vertices for each polygon cell
        verts = []
        for i in range(pv_obj.n_cells):
            cell = pv_obj.get_cell(i)
            # cell.point_ids gives the indices of the vertices that make up this cell
            verts.append(pts_2d[cell.point_ids])

        # Create a Matplotlib PolyCollection
        collection = PolyCollection(
            verts,
            array=cell_values,
            cmap="viridis",
            edgecolors="black",
            linewidths=0.2,  # Adjust for mesh density
        )

        ax.add_collection(collection)
        ax.autoscale_view()

        # Add a colorbar
        cbar = plt.colorbar(collection, ax=ax)
        cbar.set_label(field_name)

        axis_names = ["X", "Y", "Z"]
        ax.set_xlabel(f"Coordinate {axis_names[planar_axes[0]]}")
        ax.set_ylabel(f"Coordinate {axis_names[planar_axes[1]]}")
        ax.set_title(f"2D Mesh Slice: {field_name}")
        ax.set_aspect(
            "equal", "box"
        )  # Keeps the mesh aspect ratio geometrically correct

    else:
        raise ValueError(
            f"Unsupported cell dimension {max_dim}. Expected 1D or 2D slice objects."
        )


def transpose_plot(ax):
    """
    Transposes the X and Y data, limits, and labels on an existing Matplotlib axis.
    Works for both 1D line plots (Line2D) and 2D mesh slices (PolyCollections).
    """
    # 1. Swap data for 1D Line Plots
    for line in ax.lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        line.set_xdata(y_data)
        line.set_ydata(x_data)

    # 2. Swap data for 2D Mesh Slices (PolyCollections)
    for collection in ax.collections:
        paths = collection.get_paths()
        new_verts = []
        for path in paths:
            # path.vertices is an (N, 2) array. We swap column 0 and 1.
            swapped = path.vertices[:, [1, 0]]
            new_verts.append(swapped)
        collection.set_verts(new_verts)

    # 3. Swap the Axis Labels
    old_xlabel = ax.get_xlabel()
    old_ylabel = ax.get_ylabel()
    ax.set_xlabel(old_ylabel)
    ax.set_ylabel(old_xlabel)

    # 4. Swap the Axis Limits
    old_xlim = ax.get_xlim()
    old_ylim = ax.get_ylim()
    ax.set_xlim(old_ylim)
    ax.set_ylim(old_xlim)

    # 5. Force Matplotlib to redraw the canvas
    ax.figure.canvas.draw_idle()


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    sim = VTKOF("./VTK")

    # d_3d = sim.get_time_step(-1)
    # print(get_available_fields(d_3d))
    # d_3d_w = d_3d.threshold(0.001, scalars="alpha.water")
    # d_2d_w = d_3d_w.slice(normal="z", origin=(0, 0, 0.05))
    # d_2d = d_3d.slice(normal="z", origin=(0, 0, 0.05))
    # d_1d_w = d_2d_w.slice(normal="x", origin=(5, 0, 0.05))
    # d_1d_int = d_2d.integrate(normal="x", origin=(0, 0, 0.05))
    # fig, ax = plt.subplots(2, 2, constrained_layout=True)
    # plot(ax[0, 0], d_2d, "alpha.water")
    # plot(ax[1, 0], d_2d_w, "U_x")
    # plot(ax[1, 1], d_1d_w, "U_x")
    # plot(ax[0, 1], d_1d_int.ptc(), "alpha.water")
    # transpose_plot(ax[0, 0])
    # transpose_plot(ax[1, 0])
    # transpose_plot(ax[1, 1])
    # fig.savefig("slice.svg")

    # times = [1, -1]
    times = range(1, sim.size(), 10)
    timeline = np.zeros(len(times) + 1)
    beta = np.zeros(len(times) + 1)
    beta[0] = 1.0

    def compute_beta(d_2d_w):
        centers, ux, length = d_2d_w.to_np(
            "U_x", normal="x", origin=(0, 0, 0.05), use_point_data=True
        )
        _, alpha, _ = d_2d_w.to_np(
            "alpha.water", normal="x", origin=(0, 0, 0.05), use_point_data=True
        )
        idx = 2
        beta = 0
        # print(dist)
        h = np.sum(length[idx])
        # h = d_2d_w.integrate(normal="x", origin=(0, 0, 0.05)).point_data["alpha.water"][
        #    0
        # ]
        mean_u_sq = (1 / h * np.sum(alpha[idx] * ux[idx] * length[idx])) ** 2
        sq_u_mean = 1 / h * np.sum(alpha[idx] * ux[idx] * ux[idx] * length[idx])
        beta = sq_u_mean / mean_u_sq
        return beta

    for i, s in enumerate(times):
        d_3d = sim.get_time_step(s)
        d_3d_w = d_3d.threshold(0.001, scalars="alpha.water")
        d_2d_w = d_3d_w.slice(normal="z", origin=(0, 0, 0.05))
        timeline[i + 1] = sim.get_time(s)
        beta[i + 1] = compute_beta(d_2d_w)

    beta_final = beta[-1]
    beta_init = beta[0]

    def f_beta(t, tau):
        return beta_final + (beta_init - beta_final) * np.exp(-t / tau)

    from scipy.optimize import curve_fit

    popt, pcov = curve_fit(f_beta, timeline, beta)
    print(popt, pcov)

    fig, ax = plt.subplots()
    ax.plot(timeline, beta, "*-")
    ax.plot(timeline, f_beta(timeline, popt))
    fig.savefig("beta.svg")
