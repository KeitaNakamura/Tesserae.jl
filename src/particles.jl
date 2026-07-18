abstract type SamplingAlgorithm end

"""
    GridSampling(spacing = 1/2)

The generated particles are aligned with the grid using uniform spacing.
Setting `spacing = 1/η` will produce `η^dim` particles per cell, where `dim` is the problem dimension.
"""
@kwdef struct GridSampling{T} <: SamplingAlgorithm
    spacing :: T = 1/2
end

function generate_points(alg::GridSampling, mesh::CartesianMesh{dim, T}) where {dim, T}
    l = T(alg.spacing) * spacing(mesh)
    domain = tuple.(Tuple(get_xmin(mesh)), Tuple(get_xmax(mesh)))
    axis((xmin,xmax)) = (xmin+l/2):l:(xmax)
    vec(CartesianMesh(axis.(domain)...))
end

"""
    PoissonDiskSampling(spacing = 1/2, rng = Random.default_rng())

The particles are generated based on the Poisson disk sampling.
The `spacing` parameter is used to produce a similar number of particles as are generated with [`GridSampling`](@ref).
"""
@kwdef struct PoissonDiskSampling{T, RNG} <: SamplingAlgorithm
    spacing  :: T    = 1/2
    rng      :: RNG  = Random.default_rng()
    threaded :: Bool = rng isa Random.TaskLocalRNG ? true : false
end

# Determine minimum distance between particles for Poisson disk sampling
# so that the number of generated particles is almost the same as the grid sampling.
# This empirical equation is slightly different from a previous work (https://kola.opus.hbz-nrw.de/frontdoor/deliver/index/docId/2129/file/MA_Thesis_Nilles_signed.pdf)
function poisson_disk_sampling_minimum_distance(l::Real, domain::NTuple{dim}) where {dim}
    side_lengths = map(((xmin, xmax),) -> xmax - xmin, domain)
    coefficient = 0.938001956 / 1.42825365^(1/dim) + 0.0951151 * sum(l ./ side_lengths)
    coefficient * l
end

function generate_points(alg::PoissonDiskSampling, mesh::CartesianMesh{dim, T}) where {dim, T}
    l = T(alg.spacing) * spacing(mesh)
    domain = tuple.(Tuple(get_xmin(mesh)), Tuple(get_xmax(mesh)))
    d = poisson_disk_sampling_minimum_distance(l, domain)
    reinterpret(Vec{dim, T}, poisson_disk_sampling(alg.rng, T, d, domain...; threaded=alg.threaded))
end

function _generate_particles(::Type{ParticleProp}, points::AbstractArray{<: Vec}) where {ParticleProp}
    if !(isbitstype(ParticleProp))
        error("generate_particles: the property type of grid must be `isbitstype` type")
    end
    particles = StructArray{ParticleProp}(undef, size(points))
    fillzero!(particles)
    getx(particles) .= points
    particles
end

"""
    generate_particles(ParticleProp, mesh::CartesianMesh; alg=PoissonDiskSampling())

Generate particles across the entire `mesh` domain based on the selected `alg` algorithm.
See also [`GridSampling`](@ref) and [`PoissonDiskSampling`](@ref).

The generated `particles` is a [`StructArray`](https://github.com/JuliaArrays/StructArrays.jl)
where each element is of type `ParticleProp`. The first field of `ParticleProp` is designated for
particle positions.

```jldoctest
julia> ParticleProp = @NamedTuple begin
           x  :: Vec{2, Float64}
           m  :: Float64
           V  :: Float64
           v  :: Vec{2, Float64}
           F  :: SecondOrderTensor{2, Float64, 4}
           σ  :: SymmetricSecondOrderTensor{2, Float64, 3}
       end
@NamedTuple{x::Vec{2, Float64}, m::Float64, V::Float64, v::Vec{2, Float64}, F::Tensor{Tuple{2, 2}, Float64, 2, 4}, σ::SymmetricSecondOrderTensor{2, Float64, 3}}

julia> mesh = CartesianMesh(0.5, (0,3), (0,2));

julia> particles = generate_particles(ParticleProp, mesh; alg=GridSampling());

julia> particles.F[1] = one(eltype(particles.F));

julia> particles[1]
(x = [0.125, 0.125], m = 0.0, V = 0.0, v = [0.0, 0.0], F = [1.0 0.0; 0.0 1.0], σ = [0.0 0.0; 0.0 0.0])

julia> particles.F
96-element Vector{Tensor{Tuple{2, 2}, Float64, 2, 4}}:
 [1.0 0.0; 0.0 1.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 ⋮
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]
```
"""
function generate_particles(
        ::Type{ParticleProp}, mesh::CartesianMesh;
        alg::SamplingAlgorithm=PoissonDiskSampling()) where {ParticleProp}
    points = generate_points(alg, mesh)
    particles = _generate_particles(ParticleProp, points)
    particles
end

function generate_particles(mesh::CartesianMesh{dim, T}; alg::SamplingAlgorithm=PoissonDiskSampling()) where {dim, T}
    generate_particles(@NamedTuple{x::Vec{dim, T}}, mesh; alg).x
end

"""
    generate_particles(ParticleProp, mesh::FEMesh, rule::QuadratureRule)

Generate one point-property record for each point of `rule` in every finite
element cell. The result is a [`QuadraturePoints`](@ref) matrix whose rows are
rule points and whose columns are cells.
"""
function generate_particles(::Type{ParticleProp}, mesh::FEMesh, rule::QuadratureRule) where {ParticleProp}
    QuadraturePoints(_generate_particles(ParticleProp, generate_points(rule, mesh)), rule)
end

"""
    generate_particles(ParticleProp, mesh::IGAMesh, rule::QuadratureRule)

Generate one point-property record for each point of `rule` in every nonzero
knot span. The result is a [`QuadraturePoints`](@ref) matrix whose rows are rule
points and whose columns are cells.
"""
function generate_particles(::Type{ParticleProp}, mesh::IGAMesh, rule::QuadratureRule) where {ParticleProp}
    QuadraturePoints(_generate_particles(ParticleProp, generate_points(rule, mesh)), rule)
end

"""
    QuadraturePoints(particles, rule)

A matrix of point-property records associated with a reference-cell
[`QuadratureRule`](@ref). `parent(points)` returns the underlying `StructArray`,
and [`quadrature_rule(points)`](@ref) returns the stored rule.
`view(points, :, cells)` shares the storage and rule when `cells` is `:` or an `AbstractVector`.
"""
struct QuadraturePoints{T, P <: StructArray{T, 2}, R <: QuadratureRule} <: AbstractMatrix{T}
    particles::P
    rule::R
    function QuadraturePoints{T, P, R}(particles::P, rule::R) where {T, P <: StructArray{T, 2}, R <: QuadratureRule}
        length(rule.points) == length(rule.weights) || throw(DimensionMismatch("quadrature points and weights must have the same length"))
        size(particles, 1) == length(rule.points) || throw(DimensionMismatch("the first particle dimension must equal the number of quadrature points"))
        new{T, P, R}(particles, rule)
    end
end

QuadraturePoints(particles::P, rule::R) where {T, P <: StructArray{T, 2}, R <: QuadratureRule} = QuadraturePoints{T, P, R}(particles, rule)

Base.parent(points::QuadraturePoints) = getfield(points, :particles)

"""
    quadrature_rule(points::QuadraturePoints)

Return the reference-cell quadrature rule stored by `points`.
"""
quadrature_rule(points::QuadraturePoints) = getfield(points, :rule)
Base.size(points::QuadraturePoints) = size(parent(points))
Base.IndexStyle(::Type{<: QuadraturePoints{T, P}}) where {T, P} = IndexStyle(P)
Base.propertynames(points::QuadraturePoints, private::Bool=false) = propertynames(parent(points), private)
@inline Base.getproperty(points::QuadraturePoints, name::Symbol) = getproperty(parent(points), name)
@inline Base.getindex(points::QuadraturePoints, i::Int) = getindex(parent(points), i)
@inline Base.getindex(points::QuadraturePoints, i::Int, j::Int) = getindex(parent(points), i, j)
@inline Base.setindex!(points::QuadraturePoints, value, i::Int) = setindex!(parent(points), value, i)
@inline Base.setindex!(points::QuadraturePoints, value, i::Int, j::Int) = setindex!(parent(points), value, i, j)
@inline Base.view(points::QuadraturePoints, ::Colon, cells::Union{Colon, AbstractVector}) = QuadraturePoints(view(parent(points), :, cells), quadrature_rule(points))
Base.copy(points::QuadraturePoints) = QuadraturePoints(copy(parent(points)), quadrature_rule(points))
StructArrays.components(points::QuadraturePoints) = StructArrays.components(parent(points))

function _check_quadrature_rule(::QuadratureRule{F, qdim}, mesh::FEMesh) where {F, qdim}
    F === _reference_cell_family(cellshape(mesh)) || throw(ArgumentError("quadrature rule and FEM mesh must use the same reference-cell family"))
    qdim == get_dimension(cellshape(mesh)) || throw(DimensionMismatch("quadrature-rule and FEM reference dimensions must match"))
end

function _check_quadrature_rule(::QuadratureRule{F, qdim}, ::IGAMesh{dim, pdim}) where {F, qdim, dim, pdim}
    F === _tensor_product_family(Val(pdim)) || throw(ArgumentError("quadrature rule and IGA mesh must use the same reference-cell family"))
    qdim == pdim || throw(DimensionMismatch("quadrature-rule and IGA parametric dimensions must match"))
end

function generate_points(rule::QuadratureRule, mesh::Union{FEMesh, IGAMesh})
    _check_quadrature_rule(rule, mesh)
    qpts = rule.points
    points = Matrix{eltype(mesh)}(undef, length(qpts), ncells(mesh))
    for cell in cells(mesh)
        for (i, qpt) in enumerate(qpts)
            points[i,cell] = cell_point(mesh, cell, qpt)
        end
    end
    points
end

function cell_point(mesh::FEMesh, cell, qpt)
    indices = supportnodes(mesh, cell)
    N = value(cellshape(mesh), qpt)
    sum(N .* mesh[indices])
end

function cell_point(mesh::IGAMesh, cell, qpt)
    patch = patches(mesh, cell.patch)
    ξ = span_point(patch, cell.span, qpt)
    N, _ = iga_basis_values_and_gradients(patch, cell.span, ξ)
    indices = supportnodes(mesh, cell)
    R = geometry_basis_values(N, mesh.weights, indices)
    sum(R .* mesh[indices])
end

geometry_basis_values(N, ::Nothing, indices) = N
function geometry_basis_values(N, weights::AbstractVector, indices)
    w = weights[indices]
    W = sum(N .* w)
    map((Nᵢ, wᵢ) -> Nᵢ*wᵢ/W, N, w)
end
