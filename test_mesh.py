import pygmsh


def create_mesh(points_rectangular, points_hole, mesh_size):
    geom = pygmsh.occ.geometry.Geometry()

    # Create rectangular domain
    rectangle = geom.add_rectangle(points_rectangular[0], points_rectangular[1], mesh_size)

    # Create polygonal hole
    hole = geom.add_polygon(points_hole, mesh_size)

    # Subtract the hole from the rectangular domain
    geom.boolean_difference([(3, rectangle)], [(2, hole)], delete_first=True)

    # Generate the mesh
    mesh = pygmsh.generate_mesh(geom, dim=2)

    return mesh

if __name__ == "__main__":
    gmsh.initialize()

    # Define the rectangular domain's corners
    rectangular_corners = [(0.0, 0.0, 0.0), 5, 3]

    # Define the polygonal hole's vertices
    hole_vertices = [(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]

    # Specify mesh size
    mesh_size = 0.1

    # Generate the mesh
    mesh = create_mesh(rectangular_corners, hole_vertices, mesh_size)

    # Write the mesh to a file
    mesh.write("output.msh")
