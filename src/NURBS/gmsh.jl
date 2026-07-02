"""
    writestep(filename, net::ControlNet)

Write a curve, surface, or volume control net to a STEP file. Two-dimensional
curves and surfaces are embedded in the `z=0` plane. The method is provided by
the Gmsh extension.
"""
function writestep end

"""
    viewmesh(net; mesh_size_factor=0.5)

Open a 3D curve, surface, or volume control net in Gmsh with a generated mesh.
Volume control nets are shown by their boundary surface mesh.
"""
function viewmesh end
