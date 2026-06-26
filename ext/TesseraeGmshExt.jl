module TesseraeGmshExt

using Tesserae
using Gmsh

function Tesserae.readmsh(filename::AbstractString; gmsh_argv=String[])
    initialized = Gmsh.initialize(gmsh_argv; finalize_atexit=false)
    try
        Gmsh.gmsh.open(filename)
        nothing
    finally
        initialized && Gmsh.finalize()
    end
end

end
