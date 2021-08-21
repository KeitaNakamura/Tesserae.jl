function generate_pointstate(indomain, Point::Type, grid::Grid{dim, T}, coordinate_system = :plane_strain_if_2D; n::Int = 2) where {dim, T}
    h = gridsteps(grid) ./ n # length per particle
    allpoints = Grid(LinRange.(first.(gridaxes(grid)) .+ h./2, last.(gridaxes(grid)) .- h./2, n .* (size(grid) .- 1)))

    npoints = count(x -> indomain(x...), allpoints)
    pointstate = reinit!(StructVector{Point}(undef, npoints))

    cnt = 0
    for x in allpoints
        if indomain(x...)
            pointstate.x[cnt+=1] = x
        end
    end

    V = prod(h)
    for i in 1:npoints
        if dim == 2 && coordinate_system == :axisymmetric
            r = pointstate.x[i][1]
            pointstate.V0[i] = r * V
        else
            pointstate.V0[i] = V
        end
        pointstate.h[i] = Vec(h)
    end

    pointstate
end

function generate_pointstate(indomain, grid::Grid{dim, T}, coordinate_system = :plane_strain_if_2D; n::Int = 2) where {dim, T}
    generate_pointstate(indomain, @NamedTuple{x::Vec{dim, T}, V0::T, h::Vec{dim,T}}, grid, coordinate_system; n)
end

@generated function _mappedarray(f, x::V, y::V) where {names, T, V <: StructVector{T, <: NamedTuple{names}}}
    exps = [:(mappedarray(f, x.$name, y.$name)) for name in names]
    quote
        StructVector{T}(($(exps...),))
    end
end
interpolate(current, prev, α) = _mappedarray((c,p) -> (1-α)*p + α*c, current, prev)
