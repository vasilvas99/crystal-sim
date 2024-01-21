import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt

import alphashape

def rotate(l, n):
    return l[-n:] + l[:-n]


points_2d = [
    (0.0, 0.0),
    (0.0, 1.0),
    (1.0, 1.0),
    (1.0, 0.0),
    (0.5, 0.25),
    (0.5, 0.75),
    (0.25, 0.5),
    (0.75, 0.5),
    (0.5, 0.5),
    # (3, 1)
]


a = alphashape.optimizealpha(points_2d)
alpha_shape = alphashape.alphashape(points_2d, a) 

verts = list(alpha_shape.exterior.coords)
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
    n = n / np.linalg.norm(n, ord=2)
    segment_normals.append(n)

vertex_normals = []
for i in range(len(segment_normals)):
    vn = (
        segment_normals[(i + 1) % len(segment_normals)]
        + segment_normals[i % len(segment_normals)]
    )

    vn = vn / np.linalg.norm(vn, ord=2)
    vertex_normals.append(vn)
vertex_normals = rotate(vertex_normals, 1)


fig, ax = plt.subplots()
ax.scatter(*zip(*verts))
ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))

# p = len(verts)
# for i in range(p):
#     plt.quiver(verts[i][0], verts[i][1], vertex_normals[i][0], vertex_normals[i][1])

plt.xlim(-1.0, 1.5)
plt.ylim(-1.0, 1.5)
plt.show()
