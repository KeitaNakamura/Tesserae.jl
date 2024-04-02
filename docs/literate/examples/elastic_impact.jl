# # Transfer schemes
#
# ```@raw html
# <video autoplay muted loop playsinline controls width="500" src="https://github.com/KeitaNakamura/Sequoia.jl/assets/16015926/adeb872b-036f-4ba8-8915-0b9c6cf331fc"/></video>
# ```
#
# In this example, the following transfer schemes are demonstrated:
# * PIC--FLIP mixed transfer[^1]
# * Affine PIC (APIC) transfer[^2]
# * Taylor PIC (TPIC) transfer[^3]
#
# The problem evolves the elastic impact between two rings, which is consistent with previous studies[^4][^5].
#
# [^1]: [Zhu, Y. and Bridson, R., 2005. Animating sand as a fluid. ACM Transactions on Graphics (TOG), 24(3), pp.965-972.](https://doi.org/10.1145/1073204.1073298)
# [^2]: [Jiang, C., Schroeder, C., Selle, A., Teran, J. and Stomakhin, A., 2015. The affine particle-in-cell method. ACM Transactions on Graphics (TOG), 34(4), pp.1-10.](https://doi.org/10.1145/2766996)
# [^3]: [Nakamura, K., Matsumura, S. and Mizutani, T., 2023. Taylor particle-in-cell transfer and kernel correction for material point method. Computer Methods in Applied Mechanics and Engineering, 403, p.115720.](https://doi.org/10.1016/j.cma.2022.115720)
# [^4]: [de Vaucorbeil, A. and Nguyen, V.P., 2020. A numerical evaluation of the material point method for slid mechanics problems.](https://doi.org/10.13140/RG.2.2.36622.28485)
# [^5]: [Huang, P., Zhang, X., Ma, S. and Huang, X., 2011. Contact algorithms for the material point method in impact and penetration simulation. International journal for numerical methods in engineering, 85(4), pp.498-517.](https://doi.org/10.1002/nme.2981)
#

using Sequoia

abstract type Transfer end
struct FLIP <: Transfer α::Float64 end
struct APIC <: Transfer end
struct TPIC <: Transfer end

function elastic_impact(transfer::Transfer = FLIP(1.0))

    ## simulation parameters
    CFL    = 0.8    # Courant number
    Δx     = 1.0e-3 # grid spacing
    t_stop = 4e-3   # simulation stops at t=t_stop

    ## material constants
    K  = 121.7e6 # Bulk modulus
    μ  = 26.1e6  # Shear modulus
    λ  = K-2μ/3  # Lame's first parameter
    ρ⁰ = 1.01e3  # initial density

    ## geometry
    L  = 0.2  # length of domain
    W  = 0.15 # width of domain
    rᵢ = 0.03 # inner radius of rings
    rₒ = 0.04 # outer radius of rings

    ## properties for grid and particles
    GridProp = @NamedTuple begin
        x   :: Vec{2, Float64}
        m   :: Float64
        m⁻¹ :: Float64
        mv  :: Vec{2, Float64}
        f   :: Vec{2, Float64}
        v   :: Vec{2, Float64}
        vⁿ  :: Vec{2, Float64}
    end
    ParticleProp = @NamedTuple begin
        x  :: Vec{2, Float64}
        m  :: Float64
        V⁰ :: Float64
        V  :: Float64
        v  :: Vec{2, Float64}
        ∇v :: SecondOrderTensor{2, Float64, 4}
        σ  :: SymmetricSecondOrderTensor{2, Float64, 3}
        F  :: SecondOrderTensor{2, Float64, 4}
        B  :: SecondOrderTensor{2, Float64, 4} # for APIC
    end

    ## background grid
    grid = generate_grid(GridProp, CartesianMesh(Δx, (-L/2,L/2), (-W/2,W/2)))

    ## particles
    particles = let
        pts = generate_particles(ParticleProp, grid.x)
        pts.V⁰ .= prod(grid.x[end]-grid.x[1]) / length(pts)

        lhs = filter(pts) do pt
            x, y = pt.x
            rᵢ^2 < (x+L/4)^2+y^2 < rₒ^2
        end
        rhs = filter(pts) do pt
            x, y = pt.x
            rᵢ^2 < (x-L/4)^2+y^2 < rₒ^2
        end

        ## set initial velocity
        @. lhs.v =  Vec(30, 0)
        @. rhs.v = -Vec(30, 0)

        [lhs; rhs]
    end

    @. particles.V = particles.V⁰
    @. particles.m = ρ⁰ * particles.V⁰
    @. particles.F = one(particles.F)
    @show length(particles)

    ## use quadratic B-spline
    mpvalues = map(eachindex(particles)) do p
        MPValues(Vec{2, Float64}, QuadraticBSpline())
    end

    ## material model (neo-Hookean)
    function caucy_stress(F)
        b = F ⋅ F'
        J = det(F)
        (μ*(b-I) + λ*log(J)*I) / J
    end

    ## outputs
    outdir = mkpath(joinpath("output", "elastic_impact"))
    pvdfile = joinpath(outdir, "paraview")
    closepvd(openpvd(pvdfile)) # create file

    t = 0.0
    step = 0
    fps = 12e3
    savepoints = collect(LinRange(t, t_stop, round(Int, t_stop*fps)+1))

    Sequoia.@showprogress while t < t_stop

        ## calculate timestep based on the wave speed of elastic material
        Δt = CFL * spacing(grid) / maximum(LazyRows(particles)) do pt
            ρ = pt.m / pt.V
            vc = √((λ+2μ) / ρ)
            vc + norm(pt.v)
        end

        ## update MPValues
        for p in eachindex(particles, mpvalues)
            update!(mpvalues[p], LazyRow(particles, p), grid.x)
        end

        if transfer isa FLIP
            @P2G grid=>i particles=>p mpvalues=>ip begin
                m[i]  = @∑ N[ip] * m[p]
                mv[i] = @∑ N[ip] * m[p] * v[p]
                f[i]  = @∑ -V[p] * σ[p] ⋅ ∇N[ip]
            end
        elseif transfer isa APIC
            local Dₚ⁻¹ = inv(1/4 * Δx^2 * I)
            @P2G grid=>i particles=>p mpvalues=>ip begin
                m[i]  = @∑ N[ip] * m[p]
                mv[i] = @∑ N[ip] * m[p] * (v[p] + B[p] ⋅ Dₚ⁻¹ ⋅ (x[i] - x[p]))
                f[i]  = @∑ -V[p] * σ[p] ⋅ ∇N[ip]
            end
        elseif transfer isa TPIC
            @P2G grid=>i particles=>p mpvalues=>ip begin
                m[i]  = @∑ N[ip] * m[p]
                mv[i] = @∑ N[ip] * m[p] * (v[p] + ∇v[p] ⋅ (x[i] - x[p]))
                f[i]  = @∑ -V[p] * σ[p] ⋅ ∇N[ip]
            end
        end

        @. grid.m⁻¹ = inv(grid.m) * !iszero(grid.m)
        @. grid.vⁿ = grid.mv * grid.m⁻¹
        @. grid.v  = grid.vⁿ + Δt * grid.f * grid.m⁻¹

        if transfer isa FLIP
            local α = transfer.α
            @G2P grid=>i particles=>p mpvalues=>ip begin
                v[p]  = @∑ ((1-α)*v[i] + α*(v[p] + (v[i]-vⁿ[i]))) * N[ip]
                ∇v[p] = @∑ v[i] ⊗ ∇N[ip]
                x[p] += @∑ Δt * v[i] * N[ip]

            end
        elseif transfer isa APIC
            @G2P grid=>i particles=>p mpvalues=>ip begin
                v[p]  = @∑ v[i] * N[ip]
                ∇v[p] = @∑ v[i] ⊗ ∇N[ip]
                B[p]  = @∑ v[i] ⊗ (x[i]-x[p]) * N[ip]
                x[p] += Δt * v[p]
            end
        elseif transfer isa TPIC
            @G2P grid=>i particles=>p mpvalues=>ip begin
                v[p]  = @∑ v[i] * N[ip]
                ∇v[p] = @∑ v[i] ⊗ ∇N[ip]
                x[p] += Δt * v[p]
            end
        end

        ## update other particle properties
        for pt in LazyRows(particles)
            ∇u = Δt * pt.∇v
            F = (I + ∇u) ⋅ pt.F
            σ = caucy_stress(F)
            pt.σ = σ
            pt.F = F
            pt.V = det(F) * pt.V⁰
        end

        t += Δt
        step += 1

        if t > first(savepoints)
            popfirst!(savepoints)
            openpvd(pvdfile; append=true) do pvd
                openvtm(string(pvdfile, step)) do vtm
                    function stress3x3(F)
                        z = zero(Mat{2,1})
                        F3x3 = [F  z
                                z' 1]
                        caucy_stress(F3x3)
                    end
                    openvtk(vtm, particles.x) do vtk
                        vtk["velocity"] = particles.v
                        vtk["von Mises"] = @. vonmises(stress3x3(particles.F))
                    end
                    openvtk(vtm, grid.x) do vtk
                        vtk["velocity"] = grid.v
                    end
                    pvd[t] = vtm
                end
            end
        end
    end
    norm(mean(particles.x)) #src
end

## check the result                         #src
using Test                                  #src
if @isdefined(RUN_TESTS) && RUN_TESTS       #src
    @test elastic_impact(FLIP(0.0))  < 1e-3 #src
    @test elastic_impact(FLIP(1.0))  < 1e-3 #src
    @test elastic_impact(FLIP(0.99)) < 1e-3 #src
    @test elastic_impact(APIC())     < 1e-3 #src
    @test elastic_impact(TPIC())     < 1e-3 #src
end                                         #src
