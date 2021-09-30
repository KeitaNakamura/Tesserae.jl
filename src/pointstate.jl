function generate_pointstate(indomain, Point::Type, grid::Grid{dim, T}, coord_system::CoordinateSystem = DefaultCoordinateSystem(); n::Int = 2) where {dim, T}
    h = gridsteps(grid) ./ n # length per particle
    allpoints = Grid(@. LinRange(
        first($gridaxes(grid)) + h/2,
        last($gridaxes(grid))  - h/2,
        n * ($size(grid) - 1)
    ))

    npoints = count(x -> indomain(x...), allpoints)
    pointstate = reinit!(StructVector{Point}(undef, npoints))

    if :x in propertynames(pointstate)
        cnt = 0
        for x in allpoints
            if indomain(x...)
                @inbounds pointstate.x[cnt+=1] = x
            end
        end
    end
    if :V0 in propertynames(pointstate)
        V = prod(h)
        if dim == 2 && coord_system isa Axisymmetric
            @. pointstate.V0 = getindex(pointstate.x, 1) * V
        else
            @. pointstate.V0 = V
        end
    end
    if :side_length in propertynames(pointstate)
        pointstate.h .= Vec(h)
    end
    if :index in propertynames(pointstate)
        pointstate.index .= 1:npoints
    end

    pointstate
end

function generate_pointstate(indomain, grid::Grid{dim, T}, coord_system::CoordinateSystem = DefaultCoordinateSystem(); n::Int = 2) where {dim, T}
    generate_pointstate(indomain, @NamedTuple{x::Vec{dim, T}, V0::T, side_length::Vec{dim, T}, index::Int}, grid, coord_system; n)
end
