import pygmsh
import gmsh
import meshio
import matplotlib.pyplot as plt
import numpy as np
from shapely import Polygon
from shapely.plotting import plot_polygon
from dataclasses import dataclass


@dataclass
class PhysicalGroup:
    id: int
    dim_tag: int


RES = 0.1
# Channel parameters
L = 2
H = 2


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


def gen_rectangular_dom_with_poly_hole(mesh_file, resolution, l, h, poly_points):
    geom = pygmsh.geo.Geometry()
    with geom as model:
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

        geom.generate_mesh(dim=2)
        gmsh.write(mesh_file)
        gmsh.clear()


def get_physical_group(mesh, physical_name: str):
    data = mesh.field_data[physical_name]
    return PhysicalGroup(data[0], data[1])


def extract_physical_group_nodes(mesh, physical_name):
    group = get_physical_group(mesh, physical_name)
    # ofc this is non-exhaustive, but good enough...
    cell_type = ""
    if group.dim_tag == 2:
        cell_type = "triangle"
    elif group.dim_tag == 1:
        cell_type = "line"
    else:
        raise ValueError(f"Don't know how to handle {group} yet")

    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    physical_group_nodes = mesh.points[cells[cell_data == group.id]]
    physical_group_nodes = physical_group_nodes.reshape(-1, 3)[:, :2]

    return physical_group_nodes


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


def poly_as_gmsh_data(poly: Polygon):
    verts_array = np.array(poly.exterior.coords)
    verts_array = verts_array[: verts_array.shape[0] - 1]
    res = np.zeros((verts_array.shape[0], verts_array.shape[1] + 1))
    res[:, :-1] = verts_array
    return res


def make_polygon_valid(poly: Polygon):
    if poly.is_valid:
        return poly
    res = poly.buffer(0)  # remove self-interserctions
    res = Polygon(res.exterior.coords)  # remove holes
    return res


def get_boundary_with_normals(mesh, boundary_phys_group):
    bound_points = extract_physical_group_nodes(mesh, boundary_phys_group)

    # Get the normals
    bound_polygon = Polygon(bound_points)
    bound_polygon = make_polygon_valid(bound_polygon)
    bound_poly_verts = np.array(bound_polygon.exterior.coords)
    bound_poly_verts = bound_poly_verts[: bound_poly_verts.shape[0] - 1]
    bound_vert_normals = calc_poly_vert_normals(bound_poly_verts)

    assert (
        bound_vert_normals.shape == bound_poly_verts.shape
    ), f"{bound_vert_normals.shape=} != {bound_poly_verts.shape=}"

    return bound_poly_verts, bound_vert_normals


gen_rectangular_dom_with_poly_hole("tagged_hole.msh", RES, L, H, INITIAL_POLY_HOLE)
m = meshio.read("tagged_hole.msh")
verts, vert_normals = get_boundary_with_normals(m, "Hole")

# offset boundary nodes
new_verts = verts + 0.6 * vert_normals
new_poly = make_polygon_valid(Polygon(new_verts))
new_poly = new_poly.simplify(0.06, preserve_topology=True)  # Douglas-Peucker decimation

fig, ax = plt.subplots()
plot_polygon(new_poly, ax=ax)
plt.show()


gen_rectangular_dom_with_poly_hole(
    "tagged_hole_2.msh", RES, L, H, poly_as_gmsh_data(new_poly)
)
