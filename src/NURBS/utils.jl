const linear = 1
const quadratic = 2
const cubic = 3

@generated function dropat(entries::Tuple{Vararg{Any, N}}, index::Int) where {N}
    branches = map(1:N) do i
        kept = map(j -> :(entries[$j]), filter(!=(i), 1:N))
        :(index == $i && return tuple($(kept...)))
    end
    quote
        $(branches...)
        throw(ArgumentError("index must be between 1 and tuple length"))
    end
end

function check_parametric_direction(direction::Int, pdim::Integer)
    1 ≤ direction ≤ pdim || throw(ArgumentError("direction must be between 1 and the parametric dimension"))
    nothing
end

"""
    map_fibers(sweep, values, direction, n_new)

View `values` as a matrix whose columns are the tensor-product fibers along
`direction`, call `sweep(columns_new, columns_old)` to fill `n_new` rows per
fiber, and fold the result back into an array of the original shape.

`sweep` must be a separate function, not inlined code: `PermutedDimsArray`
carries the permutation as a type parameter, so a runtime `direction` leaves
`columns_old` non-inferable. Passing it as an argument specializes `sweep` on
the concrete type; written inline, every element access inside the fiber loop
is a dynamic dispatch.
"""
function map_fibers(sweep, values::Array{S, N}, direction::Int, n_new::Int) where {S, N}
    perm = ntuple(Val(N)) do i
        i == 1 && return direction
        ifelse(i ≤ direction, i - 1, i)
    end
    columns_old = reshape(PermutedDimsArray(values, perm), size(values, direction), :)
    columns_new = similar(columns_old, S, n_new, size(columns_old, 2))
    sweep(columns_new, columns_old)
    dims_new = ntuple(i -> i == 1 ? n_new : size(values, perm[i]), Val(N))
    permutedims(reshape(columns_new, dims_new), invperm(perm))
end
