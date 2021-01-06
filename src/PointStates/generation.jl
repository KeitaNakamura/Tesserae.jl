function generate_pointstates(indomain, grid::AbstractGrid{dim, T}, coordinate_system = :plane_strain_if_2D; n::Int = 2) where {dim, T}
    h = gridsteps(grid) ./ n # length per particle
    allpoints = Grid(StepRangeLen.(first.(gridaxes(grid)) .+ h./2, h, n .* (size(grid) .- 1) .- 1))
    npoints = count(x -> indomain(x...), allpoints)
    xₚ = pointstate(Vec{dim, T}, npoints)
    Vₚ = pointstate(T, npoints)
    hₚ = pointstate(Vec{dim, T}, npoints)
    i = 0
    for x in allpoints
        if indomain(x...)
            xₚ[i+=1] = x
        end
    end
    V = prod(h)
    for i in 1:npoints
        if dim == 2 && coordinate_system == :axisymmetric
            r = xₚ[i][1]
            Vₚ[i] = r * V
        else
            Vₚ[i] = V
        end
        hₚ[i] = Vec(h)
    end
    (; xₚ, Vₚ, hₚ)
end
