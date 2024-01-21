import pygmsh
import gmsh
import meshio
import matplotlib.pyplot as plt
import numpy as np
from shapely import Polygon
from shapely.plotting import plot_polygon

resolution = 0.01
# Channel parameters
L = 2
H = 2

poly_points = [
    [0.5, 0.25, 0.0],
    [0.0, 0.0, 0.0],
    [0.25, 0.5, 0.0],
    [0.0, 1.0, 0.0],
    [0.5, 0.75, 0.0],
    [1.0, 1.0, 0.0],
    [0.75, 0.5, 0.0],
    [1.0, 0.0, 0.0],
]


geom = pygmsh.geo.Geometry()
with geom as model:
    hole = model.add_polygon(poly_points, mesh_size=resolution)

    points = [
        model.add_point((-L, -L, 0), mesh_size=resolution),
        model.add_point((L, -H, 0), mesh_size=5 * resolution),
        model.add_point((L, H, 0), mesh_size=5 * resolution),
        model.add_point((-L, H, 0), mesh_size=resolution),
    ]

    channel_lines = [
        model.add_line(points[i], points[i + 1]) for i in range(-1, len(points) - 1)
    ]

    channel_loop = model.add_curve_loop(channel_lines)
    plane_surface = model.add_plane_surface(channel_loop, holes=[hole.curve_loop])

    model.synchronize()

    model.add_physical([plane_surface], "Domain")
    model.add_physical(channel_lines, "Walls")
    model.add_physical(hole.curve_loop.curves, "Hole")

    geom.generate_mesh(dim=2)
    gmsh.write("tagged_hole.msh")
    gmsh.clear()

m = meshio.read("tagged_hole.msh")

cells = m.get_cells_type("line")
cell_data = m.get_cell_data("gmsh:physical", "line")
hole_points = m.points[cells[cell_data == 3]]
hole_points = hole_points.reshape(-1, 3)[:, :2]


def rotate(l, n):
    return l[-n:] + l[:-n]


processing_poly = Polygon(hole_points)
# processing_poly = processing_poly.simplify(0.05, preserve_topology=True)

verts = list(processing_poly.exterior.coords)
verts = verts[: len(verts) - 1]

segments = []
for i in range(len(verts)):
    b = np.array(verts[(i + 1) % len(verts)])
    a = np.array(verts[i % len(verts)])
    segments.append(b - a)

segment_normals = []
for i in range(len(segments)):
    dx, dy = segments[i]
    n = np.array([-dy, dx])
    norm = np.linalg.norm(n, ord=2)
    if np.isclose(norm, 0):
        n = np.array((0, 0))
    else:
        n = n / norm
    segment_normals.append(n)

vertex_normals = []
for i in range(len(segment_normals)):
    vn = (
        segment_normals[(i + 1) % len(segment_normals)]
        + segment_normals[i % len(segment_normals)]
    )

    if np.isclose(norm, 0):
        vn = np.array((0, 0))
    else:
        vn = vn / np.linalg.norm(vn, ord=2)
    vertex_normals.append(vn)
vertex_normals = rotate(vertex_normals, 1)

assert(len(verts) == len(vertex_normals))

fig, ax = plt.subplots()
ax.scatter(*zip(*verts))
plot_polygon(processing_poly, ax=ax)


p = len(verts)
for i in range(p):
    plt.quiver(verts[i][0], verts[i][1], vertex_normals[i][0], vertex_normals[i][1])

# print(verts)

plt.xlim(-1.0, 1.5)
plt.ylim(-1.0, 1.5)
plt.show()
