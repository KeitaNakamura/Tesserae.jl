module MarbleGeometricObjectsExt

using Marble, GeometricObjects

using Marble: check_grid, check_particles

## particle_to_grid! ##

function Marble.particle_to_grid!(::Val{(:g,)}, grid::Grid, particles::Particles, space::MPSpace, geo::Geometry)
    function gap_function(pt)
        d₀ = pt.l / 2
        dₚ = distance(geo, pt.x, d₀)
        dₚ === nothing && return nothing
        d₀*normalize(dₚ) - dₚ
    end
    particle_to_grid!(Val((:g,)), grid, particles, space, gap_function)
end

function Marble.particle_to_grid!(::Val{(:fₙ,)}, grid::Grid, particles::Particles, space::MPSpace, geo::Geometry, kₙ::Real)
    check_grid(grid, space)
    check_particles(particles, space)

    mask = falses(size(grid))
    @inbounds for p in 1:length(particles)
        pt = LazyRow(particles, p)
        d₀ = pt.l / 2
        dₚ = distance(geo, pt.x, d₀)

        dₚ === nothing && continue

        gₚ = d₀*normalize(dₚ) - dₚ
        fₙₚ = -kₙ * gₚ

        mp = values(space, p)
        gridindices = neighbornodes(mp, grid)
        @simd for j in CartesianIndices(gridindices)
            i = gridindices[j]
            N = mp.N[j]
            grid.fₙ[i] += N*fₙₚ
            mask[i] = true
        end
    end

    findall(mask)
end

end # module