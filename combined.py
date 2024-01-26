from typing import Any
import gmsh
import pygmsh
import ufl
from dolfinx import fem, geometry, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from dolfinx.io import gmshio
from mpi4py import MPI
from petsc4py import PETSc
from scipy.interpolate import LinearNDInterpolator
import numpy as np
import matplotlib.pyplot as plt
from shapely import Polygon
from shapely.plotting import plot_polygon


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
MESH_MODEL_RANK = 0

RES = 0.1
# Channel parameters
L = 2
H = 2


class PiecewiseLinearInterpolator2D:
    def __init__(self, x, y, z):
        self.interp = LinearNDInterpolator(np.column_stack([x, y]), z)

    def __call__(self, x):
        x = np.array([x[0], x[1]])
        return np.apply_along_axis(self.interp, axis=0, arr=x)


INITIAL_POLY_HOLE = [
    [0.5, 0.25, 0.0],
    [0.0, 0.0, 0.0],
    [0.25, 0.5, 0.0],
    [0.0, 1.0, 0.0],
    [0.5, 0.75, 0.0],
    [1.0, 1.0, 0.0],
    [0.75, 0.5, 0.0],
    [1.0, 0.0, 0.0],
]


# def initial_condition(x, a=5):
#     return 10 * np.exp(-a * (x[0] ** 2 + x[1] ** 2))

def initial_condition(x):
    B = 5
    A = 1
    return B / A + 0.1 * A * np.random.standard_normal(size=x.shape[1])



def gen_rectangular_dom_with_poly_hole(resolution, l, h, poly_points):
    geom = pygmsh.geo.Geometry()
    model = geom.__enter__()
    hole = model.add_polygon(poly_points, mesh_size=resolution)

    points = [
        model.add_point((-l, -l, 0), mesh_size=resolution),
        model.add_point((l, -h, 0), mesh_size=5 * resolution),
        model.add_point((l, h, 0), mesh_size=5 * resolution),
        model.add_point((-l, h, 0), mesh_size=resolution),
    ]

    channel_lines = [
        model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
    ]

    channel_loop = model.add_curve_loop(channel_lines)
    plane_surface = model.add_plane_surface(channel_loop, holes=[hole.curve_loop])

    model.synchronize()
    model.add_physical([plane_surface], "Domain")
    model.add_physical(channel_lines, "Walls")
    model.add_physical(hole.curve_loop.curves, "Hole")  # 3

    model.generate_mesh(dim=2)
    model.synchronize()
    return gmsh.model


def generate_new_domain(resolution, l, h, poly_points):
    m = None
    if rank == MESH_MODEL_RANK:
        m = gen_rectangular_dom_with_poly_hole(resolution, l, h, poly_points)

    domain, cell_markers, facet_markers = gmshio.model_to_mesh(
        m, MPI.COMM_WORLD, MESH_MODEL_RANK, gdim=2
    )

    if rank == MESH_MODEL_RANK:
        gmsh.finalize()
    return domain, cell_markers, facet_markers


def run_diffusion(domain, initial_condition, diff_coef, delta_t):
    V = fem.FunctionSpace(domain, ("Lagrange", 1))
    u_n = fem.Function(V)
    u_n.name = "u_n"
    u_n.interpolate(initial_condition)

    uh = fem.Function(V)
    uh.name = "uh"
    uh.interpolate(initial_condition)

    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    f = fem.Constant(domain, PETSc.ScalarType(0))
    a = (
        u * v * ufl.dx
        + diff_coef * delta_t * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    )
    L = (u_n + delta_t * f) * v * ufl.dx

    bilinear_form = fem.form(a)
    linear_form = fem.form(L)

    A = assemble_matrix(bilinear_form)
    A.assemble()
    b = create_vector(linear_form)

    solver = PETSc.KSP().create(domain.comm)
    solver.setOperators(A)
    solver.setType(PETSc.KSP.Type.PREONLY)
    solver.getPC().setType(PETSc.PC.Type.LU)

    # Update the right hand side reusing the initial vector
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

    # Solve linear problem
    solver.solve(b, uh.vector)
    uh.x.scatter_forward()

    return V, uh


def calculate_points_at_proc(uh, domain, points):
    bb_tree = geometry.bb_tree(domain, domain.topology.dim)

    # Choose one of the cells that contains the point
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, points)
    cells = []
    for i, _point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            cells.append(colliding_cells.links(i)[0])
    res = np.squeeze(uh.eval(points, cells))
    assert (
        np.isfinite(np.isnan(res)).shape == res.shape
    ), "Non-finite value in interpolated result"

    return np.c_[points[:, 0], points[:, 1], res]


def sort_coordinates_counterclockwise(list_of_xy_coords):
    cx, cy = list_of_xy_coords.mean(0)
    x, y = list_of_xy_coords.T
    angles = np.arctan2(x - cx, y - cy)
    indices = np.argsort(angles)
    return indices, list_of_xy_coords[indices]


def normalize_stacked_vectors(stacked_vectors):
    norms = np.linalg.norm(stacked_vectors, axis=1)
    zero_norms = np.isclose(norms, 0)
    stacked_vectors[zero_norms] = (0, 0)  # for numerical stability, set to 0
    stacked_vectors[~zero_norms] /= norms[~zero_norms, None]  # otherwise, normalize


def calc_poly_vert_normals(vertices):
    vertices = np.array(vertices)
    segments = np.roll(vertices, shift=-1, axis=0) - vertices

    segment_normals = np.column_stack((-segments[:, 1], segments[:, 0]))
    normalize_stacked_vectors(segment_normals)

    vertex_normals = np.roll(segment_normals, shift=-1, axis=0) + segment_normals
    normalize_stacked_vectors(vertex_normals)

    return np.roll(vertex_normals, shift=1, axis=0)


def make_polygon_valid(poly: Polygon):
    if poly.is_valid:
        return poly
    res = poly.buffer(0)  # remove self-interserctions
    res = Polygon(res.exterior.coords)  # remove holes
    return res


def poly_as_gmsh_data(poly: Polygon):
    verts_array = np.array(poly.exterior.coords)
    verts_array = verts_array[: verts_array.shape[0] - 1]
    res = np.zeros((verts_array.shape[0], verts_array.shape[1] + 1))
    res[:, :-1] = verts_array
    return res


def main(diff_coef=0.5, delta_t=0.01, prop_coef=0.1):
    domain, _cell_markers, facet_markers = generate_new_domain(
        RES, L, H, INITIAL_POLY_HOLE
    )
    V, uh = run_diffusion(
        domain, initial_condition, diff_coef=diff_coef, delta_t=delta_t
    )

    if rank == 0:
        import os
        import shutil

        if os.path.isdir("plots"):
            shutil.rmtree("plots")
        os.mkdir("plots")

    for i in range(200):
        res = calculate_points_at_proc(uh, domain, domain.geometry.x)
        comm.barrier()
        result_all_threads = comm.allgather(res)
        final = np.row_stack(result_all_threads)
        interp = PiecewiseLinearInterpolator2D(final[:, 0], final[:, 1], final[:, 2])

        # now lets update the mesh
        
        # first we evaluate the solution at the hole points
        hole_dofs_proc = fem.locate_dofs_topological(
            V, domain.topology.dim - 1, facet_markers.find(3)
        )
        mesh_coords_proc = V.tabulate_dof_coordinates()
        hole_coords_proc = mesh_coords_proc[hole_dofs_proc]
        hole_values_proc = calculate_points_at_proc(uh, domain, hole_coords_proc)

        comm.barrier()
        hole_values_all_threads = comm.allgather(hole_values_proc)
        hole_all_vals = np.row_stack(hole_values_all_threads)
        boundary_x = hole_all_vals[:, 0]
        boundary_y = hole_all_vals[:, 1]
        boundary_val = hole_all_vals[:, 2]
        
        p = np.column_stack([boundary_x, boundary_y])
        indices, p = sort_coordinates_counterclockwise(p)
        boundary_poly = Polygon(p)
        verts = boundary_poly.exterior.coords
        verts = verts[: len(verts) - 1]
        vert_normals = calc_poly_vert_normals(verts)  # outer normals

        new_verts = verts + prop_coef * delta_t * boundary_val[indices][
            :, np.newaxis
        ] * (vert_normals - np.max(vert_normals)) / np.max(vert_normals)
        new_poly = make_polygon_valid(Polygon(new_verts))
        new_poly = new_poly.simplify(
            0.04, preserve_topology=True
        )  # Douglas-Peucker decimation

        if rank == 0:
            fig, ax = plt.subplots()
            ax.scatter(boundary_x, boundary_y, color="r")
            ax.set_xlim((-L, L))
            ax.set_ylim((-H, H))
            plot_polygon(new_poly, ax=ax)
            plt.title(f"T = {i*delta_t}, D = {diff_coef}")
            plt.savefig(f"plots/{i}.png")
            plt.close()

        domain, _cell_markers, facet_markers = generate_new_domain(
            RES, L, H, poly_as_gmsh_data(new_poly)
        )

        V, uh = run_diffusion(domain, interp, diff_coef=diff_coef, delta_t=delta_t)


if __name__ == "__main__":
    main()
