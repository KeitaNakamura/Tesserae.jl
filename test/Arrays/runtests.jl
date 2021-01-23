module ArraysTest

using Jams.Arrays
using Test, LinearAlgebra

@testset "Jams.Arrays" begin
    include("FillArray.jl")
    include("List.jl")
    include("ScalarMatrix.jl")
    include("SparseMatrixCOO.jl")
end

end
