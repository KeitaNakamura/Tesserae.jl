abstract type TransferAlgorithm end
struct DefaultTransfer <: TransferAlgorithm end
# classical
struct FLIP  <: TransferAlgorithm end
struct PIC   <: TransferAlgorithm end
# affine transfer
struct AFLIP <: TransferAlgorithm end
struct APIC  <: TransferAlgorithm end
# Taylor transfer
struct TFLIP <: TransferAlgorithm end
struct TPIC  <: TransferAlgorithm end

const AffineGroup = Union{AFLIP, APIC}
const TaylorGroup = Union{TFLIP, TPIC}
const FLIPGroup = Union{DefaultTransfer, FLIP, AFLIP, TFLIP}
const PICGroup = Union{PIC, APIC, TPIC}

###########
# helpers #
###########

function check_particles(particles::AbstractVector, space::MPSpace)
    @assert length(particles) == num_particles(space)
end

function check_grid(grid::Grid, space::MPSpace)
    @assert size(grid) == gridsize(space)
end
function check_grid(grid::Union{SpGrid, SpArray}, space::MPSpace)
    @assert size(grid) == gridsize(space)
    get_sppat(grid) !== get_gridsppat(space) &&
        error("Using different sparsity pattern between `MPSpace` and `Grid`")
end

function check_statenames(part::Tuple{Vararg{Symbol}}, all::Tuple{Vararg{Symbol}})
    isempty(setdiff!(Set(part), Set(all))) || error("unsupported state names, got $part, available names are $all")
end

################
# P2G transfer #
################

# default
function transfer!(grid::Grid, particles::Particles, space::MPSpace, dt::Real; alg::TransferAlgorithm = DefaultTransfer(), system::CoordinateSystem = NormalSystem())
    fillzero!(grid.m)
    fillzero!(grid.vⁿ)
    fillzero!(grid.v)
    transfer!(alg, system, Val((:m, :f, :mv)), @rename(grid, v=>f, vⁿ=>mv), particles, space)
    @. grid.v = ((grid.vⁿ + dt*grid.v) / grid.m) * !iszero(grid.m)
    @. grid.vⁿ = (grid.vⁿ / grid.m) * !iszero(grid.m)
    grid
end

# don't use dispatch and all transfer algorithms are writtein in this function to reduce a lot of deplicated code
function transfer!(alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, grid::Grid, particles::Particles, space::MPSpace{dim, T}) where {names, dim, T}
    check_statenames(names, (:m, :f, :mv))
    check_grid(grid, space)
    check_particles(particles, space)

    lattice = get_lattice(space)
    parallel_each_particle(space) do p
        @inbounds begin
            mp = mpvalue(space, p)

            # 100% used
            mₚ = particles.m[p]

            if :f in names
                xₚ = particles.x[p]
                Vₚ = particles.V[p]
                σₚ = particles.σ[p]
                bₚ = particles.b[p]
                Vₚσₚ = Vₚ * σₚ
                mₚbₚ = mₚ * bₚ
            end

            # grid momentum depends on transfer algorithms
            if :mv in names
                if alg isa DefaultTransfer && mp isa WLSValue
                    P = x -> value(get_basis(mp), x)
                    Cₚ = particles.C[p]
                    mₚCₚ = mₚ * Cₚ
                else
                    vₚ = particles.v[p]
                    mₚvₚ = mₚ * vₚ

                    # additional term from high order approximation
                    if alg isa AffineGroup
                        Bₚ = particles.B[p]
                        Dₚ = zero(Mat{dim, dim, T})
                        for (j, i) in enumerate(neighbornodes(space, p))
                            N = shape_value(mp, j)
                            xᵢ = lattice[i]
                            Dₚ += N*(xᵢ-xₚ)⊗(xᵢ-xₚ)
                        end
                        mₚCₚ = mₚ * Bₚ ⋅ inv(Dₚ)
                    elseif alg isa TaylorGroup
                        ∇vₚ = particles.∇v[p]
                        mₚ∇vₚ = mₚ * @Tensor(∇vₚ[1:dim, 1:dim])
                    end
                end
            end

            for (j, i) in enumerate(neighbornodes(space, p))
                N = shape_value(mp, j)
                ∇N = shape_gradient(mp, j)

                if :m in names
                    grid.m[i] += N*mₚ
                end

                if :f in names
                    fint = -stress_to_force(system, N, ∇N, xₚ, Vₚσₚ)
                    grid.f[i] += fint + N*mₚbₚ
                end

                # grid momentum depends on transfer algorithms
                if :mv in names
                    if alg isa DefaultTransfer && mp isa WLSValue
                        xᵢ = lattice[i]
                        grid.mv[i] += N*mₚCₚ⋅P(xᵢ-xₚ)
                    elseif alg isa AffineGroup
                        xᵢ = lattice[i]
                        grid.mv[i] += N*(mₚvₚ + mₚCₚ⋅(xᵢ-xₚ))
                    elseif alg isa TaylorGroup
                        xᵢ = lattice[i]
                        grid.mv[i] += N*(mₚvₚ + mₚ∇vₚ⋅(xᵢ-xₚ))
                    else
                        grid.mv[i] += N*mₚvₚ
                    end
                end
            end
        end
    end

    grid
end

@inline function stress_to_force(::Union{NormalSystem, PlaneStrain}, N, ∇N, x::Vec{2}, σ::SymmetricSecondOrderTensor{3})
    Tensorial.resizedim(σ, Val(2)) ⋅ ∇N
end
@inline function stress_to_force(::Axisymmetric, N, ∇N, x::Vec{2}, σ::SymmetricSecondOrderTensor{3})
    @inbounds Tensorial.resizedim(σ, Val(2)) ⋅ ∇N + Vec(1,0) * (σ[3,3] * N / x[1])
end
@inline function stress_to_force(::NormalSystem, N, ∇N, x::Vec{3}, σ::SymmetricSecondOrderTensor{3})
    σ ⋅ ∇N
end

################
# G2P transfer #
################

# default
function transfer!(particles::Particles, grid::Grid, space::MPSpace, dt::Real; alg::TransferAlgorithm = DefaultTransfer(), system::CoordinateSystem = NormalSystem())
    transfer!(alg, system, Val((:∇v, :x, :v)), particles, grid, space, dt)
end

function transfer!(alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, particles::Particles, grid::Grid, space::MPSpace{dim}, dt::Real) where {names, dim}
    check_statenames(names, (:∇v, :x, :v))
    check_grid(grid, space)
    check_particles(particles, space)

    @threaded for p in 1:num_particles(space)
        mp = mpvalue(space, p)

        # 100% used
        xₚ  = particles.x[p]

        # there is no difference along with algorithms for calculating `:∇v` and `:x`
        if :∇v in names
            vₚ  = zero(eltype(particles.v))
            ∇vₚ = @Tensor zero(eltype(particles.∇v))[1:dim, 1:dim]
        end

        if :x in names
            vₚ = zero(eltype(particles.v))
        end

        # particle velocity depends on transfer algorithms
        if :v in names
            if alg isa FLIPGroup
                dvₚ = zero(eltype(particles.v))
            else
                @assert alg isa PICGroup
                vₚ = zero(eltype(particles.v))
            end
            if alg isa AffineGroup
                # Bₚ is always calculated when `:v` is specified
                Bₚ = zero(eltype(particles.B))
            end
        end

        lattice = get_lattice(space)
        for (j, i) in enumerate(neighbornodes(space, p))
            N = shape_value(mp, j)
            ∇N = shape_gradient(mp, j)

            # 100% used
            vᵢ = grid.v[i]

            if :∇v in names
                ∇vₚ += vᵢ ⊗ ∇N
            end

            # use `@isdefined` to avoid complicated check
            # for `:v` in `PIC` is also calculated here
            if @isdefined vₚ
                vₚ += vᵢ * N
            end

            # particle velocity depends on transfer algorithms
            if :v in names
                if alg isa FLIPGroup
                    dvᵢ = vᵢ - grid.vⁿ[i]
                    dvₚ += N * dvᵢ
                end
                if alg isa AffineGroup
                    xᵢ = lattice[i]
                    Bₚ += N * vᵢ ⊗ (xᵢ - xₚ)
                end
            end
        end

        if :∇v in names
            particles.∇v[p] = velocity_gradient(system, xₚ, vₚ, ∇vₚ)
        end

        if :x in names
            particles.x[p] = xₚ + vₚ * dt
        end

        # particle velocity depends on transfer algorithms
        if :v in names
            if alg isa FLIPGroup
                particles.v[p] += dvₚ
            else
                @assert alg isa PICGroup
                particles.v[p] = vₚ
            end
            if alg isa AffineGroup
                # additional quantity for affine transfers
                # Bₚ is always calculated when `:v` is specified
                particles.B[p] = Bₚ
            end
        end
    end

    particles
end

# special default transfer for `WLS` interpolation
function transfer!(::DefaultTransfer, system::CoordinateSystem, ::Val{names}, particles::Particles, grid::Grid, space::MPSpace{dim, <: Any, <: WLSValue}, dt::Real) where {names, dim}
    check_statenames(names, (:∇v, :x, :v))
    check_grid(grid, space)
    :∇v in names && check_particles(particles.∇v, space)
    :x  in names && check_particles(particles.x, space)
    :v  in names && check_particles(particles.v, space)

    @threaded for p in 1:num_particles(space)
        mp = mpvalue(space, p)

        xₚ = particles.x[p]
        Cₚ = zero(eltype(particles.C))
        P = x -> value(get_basis(mp), x)

        for (j, i) in enumerate(neighbornodes(space, p))
            w = mp.w[j]
            Minv = mp.Minv
            vᵢ = grid.v[i]
            xᵢ = grid.x[i]
            Cₚ += vᵢ ⊗ (w * Minv ⋅ P(xᵢ - xₚ))
        end

        p0 = value(get_basis(mp), zero(Vec{dim, Int}))
        vₚ = Cₚ ⋅ p0

        if :∇v in names
            ∇p0 = gradient(get_basis(mp), zero(Vec{dim, Int}))
            particles.∇v[p] = velocity_gradient(system, xₚ, vₚ, Cₚ ⋅ ∇p0)
        end

        if :x in names
            particles.x[p] = xₚ + vₚ * dt
        end

        if :v in names
            particles.v[p] = vₚ
            particles.C[p] = Cₚ # always update when velocity is updated
        end
    end

    particles
end

@inline function velocity_gradient(::Union{NormalSystem, PlaneStrain}, x::Vec{2}, v::Vec{2}, ∇v::SecondOrderTensor{2})
    Tensorial.resizedim(∇v, Val(3)) # expaned entries are filled with zero
end
@inline function velocity_gradient(::Axisymmetric, x::Vec{2}, v::Vec{2}, ∇v::SecondOrderTensor{2})
    @inbounds Tensorial.resizedim(∇v, Val(3)) + @Mat([0 0 0; 0 0 0; 0 0 v[1]/x[1]])
end
@inline function velocity_gradient(::NormalSystem, x::Vec{3}, v::Vec{3}, ∇v::SecondOrderTensor{3})
    ∇v
end

##########################
# smooth_particle_state! #
##########################

@generated function safe_inv(x::Mat{dim, dim, T, L}) where {dim, T, L}
    exps = fill(:z, L-1)
    quote
        @_inline_meta
        z = zero(T)
        isapproxzero(det(x)) ? Mat{dim, dim}(inv(x[1]), $(exps...)) : inv(x)
        # Tensorial.rank(x) != dim ? Mat{dim, dim}(inv(x[1]), $(exps...)) : inv(x) # this is very slow but stable
    end
end

function smooth_particle_state!(vals::AbstractVector, xₚ::AbstractVector, Vₚ::AbstractVector, grid::Grid, space::MPSpace)
    check_grid(grid, space)
    check_particles(vals, space)
    check_particles(xₚ, space)
    check_particles(Vₚ, space)

    basis = PolynomialBasis{1}()
    fillzero!(grid.poly_coef)
    fillzero!(grid.poly_mat)

    parallel_each_particle(space) do p
        @inbounds begin
            mp = mpvalue(space, p)
            for (j, i) in enumerate(neighbornodes(space, p))
                N = shape_value(mp, j)
                P = value(basis, xₚ[p] - grid.x[i])
                VP = (N * Vₚ[p]) * P
                grid.poly_coef[i] += VP * vals[p]
                grid.poly_mat[i]  += VP ⊗ P
            end
        end
    end

    @. grid.poly_coef = safe_inv(grid.poly_mat) ⋅ grid.poly_coef

    @threaded for p in 1:num_particles(space)
        val = zero(eltype(vals))
        mp = mpvalue(space, p)
        for (j, i) in enumerate(neighbornodes(space, p))
            N = shape_value(mp, j)
            P = value(basis, xₚ[p] - grid.x[i])
            val += N * (P ⋅ grid.poly_coef[i])
        end
        vals[p] = val
    end

    vals
end
