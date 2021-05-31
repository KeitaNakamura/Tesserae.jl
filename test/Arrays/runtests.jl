module ArraysTest

using Poingr.Arrays
using Test, LinearAlgebra

@testset "Poingr.Arrays" begin
    include("FillArray.jl")
    include("List.jl")
    include("ScalarMatrix.jl")
    include("SparseMatrixCOO.jl")
end

end
