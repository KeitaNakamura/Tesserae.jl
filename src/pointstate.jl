function generate_pointstate(indomain, Point::Type, grid::Grid{dim, T}, coord_system::CoordinateSystem = DefaultCoordinateSystem(); n::Int = 2) where {dim, T}
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
        if dim == 2 && coord_system isa Axisymmetric
            r = pointstate.x[i][1]
            pointstate.V0[i] = r * V
        else
            pointstate.V0[i] = V
        end
        pointstate.h[i] = Vec(h)
    end

    pointstate
end

function generate_pointstate(indomain, grid::Grid{dim, T}, coord_system::CoordinateSystem = DefaultCoordinateSystem(); n::Int = 2) where {dim, T}
    generate_pointstate(indomain, @NamedTuple{x::Vec{dim, T}, V0::T, h::Vec{dim,T}}, grid, coord_system; n)
end
