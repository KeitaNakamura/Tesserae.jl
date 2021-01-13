module Arrays

import SparseArrays: sparse

export
# FillArray
    FillArray,
    Ones,
    Zeros,
# ScalarMatrix
    ScalarMatrix,
# SparseMatrixCOO
    SparseMatrixCOO,
    sparse

include("FillArray.jl")
include("ScalarMatrix.jl")
include("SparseMatrixCOO.jl")

end
