module Arrays

import SparseArrays: sparse

using Base: @_propagate_inbounds_meta

export
# FillArray
    FillArray,
    Ones,
    Zeros,
# ScalarMatrix
    ScalarMatrix,
# SparseMatrixCOO
    SparseMatrixCOO,
    sparse,
# List
    List,
    ListGroup

include("FillArray.jl")
include("ScalarMatrix.jl")
include("SparseMatrixCOO.jl")
include("List.jl")

end
