abstract type TransferAlgorithm end
struct DefaultTransfer <: TransferAlgorithm end
# classical
struct FLIP  <: TransferAlgorithm end
struct PIC   <: TransferAlgorithm end
# affine transfer
struct AffineTransfer{T <: Union{FLIP, PIC}} <: TransferAlgorithm end
AffineTransfer(t::TransferAlgorithm) = AffineTransfer{typeof(t)}()
const AFLIP = AffineTransfer{FLIP}
const APIC  = AffineTransfer{PIC}
# Taylor transfer
struct TaylorTransfer{T <: Union{FLIP, PIC}} <: TransferAlgorithm end
TaylorTransfer(t::TransferAlgorithm) = TaylorTransfer{typeof(t)}()
const TFLIP = TaylorTransfer{FLIP}
const TPIC  = TaylorTransfer{PIC}

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
    get_spinds(grid) !== get_gridspinds(space) &&
        error("Using different sparsity pattern between `MPSpace` and `Grid`")
end

function check_statenames(part::Tuple{Vararg{Symbol}}, all::Tuple{Vararg{Symbol}})
    isempty(setdiff!(Set(part), Set(all))) || error("unsupported state names, got $part, available names are $all")
end

################
# P2G transfer #
################

function particle_to_grid!(names::Tuple{Vararg{Symbol}}, grid::Grid, particles::Particles, space::MPSpace; alg::TransferAlgorithm = DefaultTransfer(), system::CoordinateSystem = NormalSystem(), parallel::Bool=true)
    particle_to_grid!(alg, system, Val(names), grid, particles, space; parallel)
end

function particle_to_grid!(alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, grid::Grid, particles::Particles, space::MPSpace; parallel::Bool) where {names}
    check_statenames(names, (:m, :mv, :f))
    check_grid(grid, space)
    check_particles(particles, space)
    parallel_each_particle(space; parallel) do p
        @inbounds particle_to_grid!(alg, system, Val(names), grid, LazyRow(particles, p), get_interpolation(space), values(space, p))
    end
    grid
end

# don't use dispatch and all transfer algorithms are writtein in this function to reduce a lot of deplicated code
@inline function particle_to_grid!(alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, grid::Grid, pt, itp::Interpolation, mp::SubMPValues{dim, T}) where {names, dim, T}
    @_propagate_inbounds_meta

    if :m in names
        mₚ = pt.m
    end

    if :f in names
        Vₚσₚ = pt.V * pt.σ
        if hasproperty(pt, :b)
            mₚbₚ = pt.m * pt.b
        end
        if system isa Axisymmetric
            rₚ = pt.x[1]
        end
    end

    # grid momentum depends on transfer algorithms
    if :mv in names
        if alg isa DefaultTransfer && itp isa WLS
            P = x -> value(get_basis(itp), x)
            xₚ = pt.x
            mₚCₚ = pt.m * pt.C
        else
            mₚvₚ = pt.m * pt.v

            # additional term from high order approximation
            if alg isa AffineTransfer
                xₚ = pt.x
                Dₚ = zero(Mat{dim, dim, T})
                for (j, i) in pairs(IndexCartesian(), neighbornodes(mp, grid))
                    N = mp.N[j]
                    xᵢ = grid.x[i]
                    Dₚ += N*(xᵢ-xₚ)⊗(xᵢ-xₚ)
                end
                mₚCₚ = pt.m * pt.B ⋅ inv(Dₚ)
            elseif alg isa TaylorTransfer
                xₚ = pt.x
                mₚ∇vₚ = pt.m * @Tensor(pt.∇v[1:dim, 1:dim])
            end
        end
    end

    for (j, i) in pairs(IndexCartesian(), neighbornodes(mp, grid))
        N = mp.N[j]
        ∇N = mp.∇N[j]

        if :m in names
            grid.m[i] += N*mₚ
        end

        if :f in names
            if system isa Axisymmetric
                f = -calc_fint(system, N, ∇N, Vₚσₚ, rₚ)
            else
                f = -calc_fint(system, ∇N, Vₚσₚ)
            end
            if hasproperty(pt, :b)
                f += N*mₚbₚ
            end
            grid.f[i] += f
        end

        # grid momentum depends on transfer algorithms
        if :mv in names
            xᵢ = grid.x[i]
            if alg isa DefaultTransfer && itp isa WLS
                grid.mv[i] += N*mₚCₚ⋅P(xᵢ-xₚ)
            elseif alg isa AffineTransfer
                grid.mv[i] += N*(mₚvₚ + mₚCₚ⋅(xᵢ-xₚ))
            elseif alg isa TaylorTransfer
                grid.mv[i] += N*(mₚvₚ + mₚ∇vₚ⋅(xᵢ-xₚ))
            else
                grid.mv[i] += N*mₚvₚ
            end
        end
    end
end

# 1D
@inline calc_fint(::NormalSystem, ∇N::Vec{1}, Vₚσₚ::SymmetricSecondOrderTensor{1}) = Vₚσₚ ⋅ ∇N
# plane-strain
@inline calc_fint(::Union{NormalSystem, PlaneStrain}, ∇N::Vec{2}, Vₚσₚ::SymmetricSecondOrderTensor{3}) = @Tensor(Vₚσₚ[1:2,1:2]) ⋅ ∇N
@inline calc_fint(::Union{NormalSystem, PlaneStrain}, ∇N::Vec{2}, Vₚσₚ::SymmetricSecondOrderTensor{2}) = Vₚσₚ ⋅ ∇N
# axisymmetric
@inline calc_fint(::Axisymmetric, N::Real, ∇N::Vec{2}, Vₚσₚ::SymmetricSecondOrderTensor{3}, rₚ::Real) = @Tensor(Vₚσₚ[1:2,1:2])⋅∇N + Vec(1,0)*Vₚσₚ[3,3]*N*rₚ
# 3D
@inline calc_fint(::NormalSystem, ∇N::Vec{3}, Vₚσₚ::SymmetricSecondOrderTensor{3}) = Vₚσₚ ⋅ ∇N

################
# G2P transfer #
################

function grid_to_particle!(names::Tuple{Vararg{Symbol}}, particles::Particles, grid::Grid, space::MPSpace, only_dt...; alg::TransferAlgorithm = DefaultTransfer(), system::CoordinateSystem = NormalSystem(), parallel::Bool=true)
    grid_to_particle!(alg, system, Val(names), particles, grid, space, only_dt...; parallel)
end

function grid_to_particle!(alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, particles::Particles, grid::Grid, space::MPSpace{dim}, only_dt...; parallel::Bool) where {names, dim}
    check_statenames(names, (:v, :∇v, :x))
    check_grid(grid, space)
    check_particles(particles, space)
    @threaded_inbounds parallel for p in 1:num_particles(space)
        grid_to_particle!(alg, system, Val(names), LazyRow(particles, p), grid, get_interpolation(space), values(space, p), only_dt...)
    end
    particles
end

@inline function grid_to_particle!(alg::TransferAlgorithm, system::CoordinateSystem, ::Val{names}, pt, grid::Grid, itp::Interpolation, mp::SubMPValues{dim}, only_dt...) where {names, dim}
    @_propagate_inbounds_meta

    # there is no difference along with transfer algorithms for calculating `:∇v` and `:x`
    if :∇v in names
        ∇vₚ = @Tensor zero(pt.∇v)[1:dim, 1:dim]
        if system isa Axisymmetric
            vₚ = zero(pt.v)
        end
    end

    if :x in names
        vₚ = zero(pt.v)
    end

    # particle velocity depends on transfer algorithms
    if :v in names
        if alg isa FLIPGroup
            dvₚ = zero(pt.v)
        else
            @assert alg isa PICGroup
            vₚ = zero(pt.v)
        end
        if alg isa AffineTransfer
            # Bₚ is always calculated when `:v` is specified
            xₚ = pt.x
            Bₚ = zero(pt.B)
        end
    end

    for (j, i) in pairs(IndexCartesian(), neighbornodes(mp, grid))
        N = mp.N[j]
        ∇N = mp.∇N[j]

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
            if alg isa AffineTransfer
                xᵢ = grid.x[i]
                Bₚ += N * vᵢ ⊗ (xᵢ - xₚ)
            end
        end
    end

    if :∇v in names
        if system isa Axisymmetric
            pt.∇v = calc_∇v(system, typeof(pt.∇v), ∇vₚ, vₚ[1], pt.x[1])
        else
            pt.∇v = calc_∇v(system, typeof(pt.∇v), ∇vₚ)
        end
    end

    if :x in names
        dt = only(only_dt)
        pt.x += vₚ * dt
    end

    # particle velocity depends on transfer algorithms
    if :v in names
        if alg isa FLIPGroup
            pt.v += dvₚ
        else
            @assert alg isa PICGroup
            pt.v = vₚ
        end
        if alg isa AffineTransfer
            # additional quantity for affine transfers
            # Bₚ is always calculated when `:v` is specified
            pt.B = Bₚ
        end
    end
end

# special default transfer for `WLS` interpolation
@inline function grid_to_particle!(::DefaultTransfer, system::CoordinateSystem, ::Val{names}, pt, grid::Grid, itp::WLS, mp::SubMPValues{dim}, dt::Real) where {names, dim}
    @_propagate_inbounds_meta

    basis = get_basis(itp)
    P = x -> value(basis, x)
    p0 = value(basis, zero(Vec{dim, Int}))
    ∇p0 = gradient(basis, zero(Vec{dim, Int}))

    xₚ = pt.x
    Cₚ = zero(pt.C)

    for (j, i) in pairs(IndexCartesian(), neighbornodes(mp, grid))
        w = mp.w[j]
        Minv = mp.Minv[]
        vᵢ = grid.v[i]
        xᵢ = grid.x[i]
        Cₚ += vᵢ ⊗ (w * Minv ⋅ P(xᵢ - xₚ))
    end

    vₚ = Cₚ ⋅ p0

    if :∇v in names
        ∇vₚ = Cₚ ⋅ ∇p0
        if system isa Axisymmetric
            pt.∇v = calc_∇v(system, typeof(pt.∇v), ∇vₚ, vₚ[1], pt.x[1])
        else
            pt.∇v = calc_∇v(system, typeof(pt.∇v), ∇vₚ)
        end
    end

    if :x in names
        pt.x += vₚ * dt
    end

    if :v in names
        pt.v = vₚ
        pt.C = Cₚ # always update when velocity is updated
    end
end

# 1D
@inline calc_∇v(::NormalSystem, ::Type{<: SecondOrderTensor{1}}, ∇vₚ::SecondOrderTensor{1}) = ∇vₚ
# plane-strain
@inline calc_∇v(::Union{NormalSystem, PlaneStrain}, ::Type{<: SecondOrderTensor{2}}, ∇vₚ::SecondOrderTensor{2}) = ∇vₚ
@inline calc_∇v(::Union{NormalSystem, PlaneStrain}, ::Type{<: SecondOrderTensor{3}}, ∇vₚ::SecondOrderTensor{2}) = Tensorial.resizedim(∇vₚ, Val(3))
# axisymmetric
@inline calc_∇v(::Axisymmetric, ::Type{<: SecondOrderTensor{3}}, ∇vₚ::SecondOrderTensor{2}, v::Real, r::Real) = Tensorial.resizedim(∇v, Val(3)) + @Mat([0 0 0; 0 0 0; 0 0 v/r])
# 3D
@inline calc_∇v(::NormalSystem, ::Type{<: SecondOrderTensor{3}}, ∇vₚ::SecondOrderTensor{3}) = ∇vₚ

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

function smooth_particle_state!(vals::AbstractVector, xₚ::AbstractVector, Vₚ::AbstractVector, grid::Grid, space::MPSpace; parallel::Bool=true)
    check_grid(grid, space)
    check_particles(vals, space)
    check_particles(xₚ, space)
    check_particles(Vₚ, space)

    basis = PolynomialBasis{1}()
    fillzero!(grid.poly_coef)
    fillzero!(grid.poly_mat)

    parallel_each_particle(space; parallel) do p
        @inbounds begin
            mp = values(space, p)
            for (j, i) in pairs(IndexCartesian(), neighbornodes(space, grid, p))
                N = mp.N[j]
                P = value(basis, xₚ[p] - grid.x[i])
                VP = (N * Vₚ[p]) * P
                grid.poly_coef[i] += VP * vals[p]
                grid.poly_mat[i]  += VP ⊗ P
            end
        end
    end

    @. grid.poly_coef = safe_inv(grid.poly_mat) ⋅ grid.poly_coef

    @threaded_inbounds parallel for p in 1:num_particles(space)
        val = zero(eltype(vals))
        mp = values(space, p)
        for (j, i) in pairs(IndexCartesian(), neighbornodes(space, grid, p))
            N = mp.N[j]
            P = value(basis, xₚ[p] - grid.x[i])
            val += N * (P ⋅ grid.poly_coef[i])
        end
        vals[p] = val
    end

    vals
end
