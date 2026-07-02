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
