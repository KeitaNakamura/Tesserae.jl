"""
    CPDI()

A kernel for convected particle domain interpolation (CPDI) [^CPDI].
`CPDI` requires the initial particle length `l` and the deformation gradient `F`
in the particle property.
For example, in two dimensions, the property is likely to be as follows:

```jl
ParticleProp = @NamedTuple begin
    < variables... >
    l :: Float64
    F :: Mat{2, 2, Float64, 4}
end
```

[^CPDI]: [Sadeghirad, A., Brannon, R.M. and Burghardt, J., 2011. A convected particle domain interpolation technique to extend applicability of the material point method for problems involving massive deformations. International Journal for numerical methods in Engineering, 86(12), pp.1435-1456.](https://doi.org/10.1002/nme.3110)
"""
struct CPDI <: Kernel end

struct PseudoResizedVector{T} <: AbstractVector{T}
    len::Base.RefValue{Int}
    data::Vector{T}
end
PseudoResizedVector{T}(; maxlength::Int) where {T} = PseudoResizedVector(Ref(0), Vector{T}(undef, maxlength))
Base.size(x::PseudoResizedVector) = (x.len[],)
@inline maxlength(x) = length(x.data)
@inline function Base.getindex(x::PseudoResizedVector, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds x.data[i]
end
@inline function Base.setindex!(x::PseudoResizedVector, v, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds x.data[i] = v
end
@inline function Base.resize!(x::PseudoResizedVector, len::Int)
    @assert len ≤ length(x.data)
    x.len[] = len
    x
end

@generated function create_property(::Type{Vec{dim, T}}, it::CPDI; name::Val{sym}=Val(:w)) where {dim, T, sym}
    w = sym
    ∇w = Symbol(:∇, sym)
    quote
        len = 2^dim * 2^dim # maximum length
        (; $w=fill(zero(T), len), $∇w=fill(zero(Vec{dim, T}), len))
    end
end

function initial_neighboringnodes(::CPDI, mesh::CartesianMesh{dim}) where {dim}
    PseudoResizedVector{CartesianIndex{dim}}(; maxlength = 2^dim * 2^dim)
end

function update!(mp::MPValue{CPDI}, pt, mesh::CartesianMesh{1})
    xₚ = getx(pt)
    r₁ = pt.F * Vec(pt.l/2)
    x₁ = xₚ - r₁
    x₂ = xₚ + r₁
    Vₚ = 2r₁[1]

    indices = neighboringnodes_storage(mp)[]
    find_neighboringnodes_cpdi!(indices, (x₁, x₂), mesh)

    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        w₁ = value(BSpline(Linear()), x₁, mesh, i)
        w₂ = value(BSpline(Linear()), x₂, mesh, i)
        w = (w₁ + w₂) / 2
        ∇w = Vec(w₂ - w₁) / Vₚ
        set_values!(mp, ip, (w,∇w))
    end
end

function update!(mp::MPValue{CPDI}, pt, mesh::CartesianMesh{2})
    xₚ = getx(pt)
    r₁ = pt.F * Vec(pt.l/2,0)
    r₂ = pt.F * Vec(0,pt.l/2)
    x₁ = xₚ - r₁ - r₂
    x₂ = xₚ + r₁ - r₂
    x₃ = xₚ + r₁ + r₂
    x₄ = xₚ - r₁ + r₂
    Vₚ = 4 * abs(r₁ × r₂)

    indices = neighboringnodes_storage(mp)[]
    find_neighboringnodes_cpdi!(indices, (x₁, x₂, x₃, x₄), mesh)

    a = Vec(r₁[2]-r₂[2], r₂[1]-r₁[1])
    b = Vec(r₁[2]+r₂[2],-r₁[1]-r₂[1])
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        w₁ = value(BSpline(Linear()), x₁, mesh, i)
        w₂ = value(BSpline(Linear()), x₂, mesh, i)
        w₃ = value(BSpline(Linear()), x₃, mesh, i)
        w₄ = value(BSpline(Linear()), x₄, mesh, i)
        w = (w₁ + w₂ + w₃ + w₄) / 4
        ∇w = ((w₁-w₃)*a + (w₂-w₄)*b) / Vₚ
        set_values!(mp, ip, (w,∇w))
    end
end

function update!(mp::MPValue{CPDI}, pt, mesh::CartesianMesh{3})
    xₚ = getx(pt)
    r₁ = pt.F * Vec(pt.l/2,0,0)
    r₂ = pt.F * Vec(0,pt.l/2,0)
    r₃ = pt.F * Vec(0,0,pt.l/2)
    x₁ = xₚ - r₁ - r₂ - r₃
    x₂ = xₚ + r₁ - r₂ - r₃
    x₃ = xₚ + r₁ + r₂ - r₃
    x₄ = xₚ - r₁ + r₂ - r₃
    x₅ = xₚ - r₁ - r₂ + r₃
    x₆ = xₚ + r₁ - r₂ + r₃
    x₇ = xₚ + r₁ + r₂ + r₃
    x₈ = xₚ - r₁ + r₂ + r₃
    Vₚ = 8 * (r₁ × r₂) ⋅ r₃

    indices = neighboringnodes_storage(mp)[]
    find_neighboringnodes_cpdi!(indices, (x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈), mesh)

    A = @Mat [r₂[2]*r₃[3]-r₃[2]*r₂[3] r₃[2]*r₁[3]-r₁[2]*r₃[3] r₁[2]*r₂[3]-r₂[2]*r₁[3]
              r₃[1]*r₂[3]-r₂[1]*r₃[3] r₁[1]*r₃[3]-r₃[1]*r₁[3] r₂[1]*r₁[3]-r₁[1]*r₂[3]
              r₂[1]*r₃[2]-r₃[1]*r₂[2] r₃[1]*r₁[2]-r₁[1]*r₃[2] r₁[1]*r₂[2]-r₂[1]*r₁[2]]
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        w₁ = value(BSpline(Linear()), x₁, mesh, i)
        w₂ = value(BSpline(Linear()), x₂, mesh, i)
        w₃ = value(BSpline(Linear()), x₃, mesh, i)
        w₄ = value(BSpline(Linear()), x₄, mesh, i)
        w₅ = value(BSpline(Linear()), x₅, mesh, i)
        w₆ = value(BSpline(Linear()), x₆, mesh, i)
        w₇ = value(BSpline(Linear()), x₇, mesh, i)
        w₈ = value(BSpline(Linear()), x₈, mesh, i)
        w = (w₁ + w₂ + w₃ + w₄ + w₅ + w₆ + w₇ + w₈) / 8
        ∇w = A * (w₁*Vec(-1,-1,-1) + w₂*Vec(1,-1,-1) + w₃*Vec(1,1,-1) + w₄*Vec(-1,1,-1) + w₅*Vec(-1,-1,1) + w₆*Vec(1,-1,1) + w₇*Vec(1,1,1) + w₈*Vec(-1,1,1)) / Vₚ
        set_values!(mp, ip, (w,∇w))
    end
end

function find_neighboringnodes_cpdi!(indices::PseudoResizedVector, corners, mesh)
    resize!(indices, maxlength(indices)) # set maximum size
    count = 0
    @inbounds for x in corners
        count_inner = count
        for i in neighboringnodes(BSpline(Linear()), x, mesh)
            if all(j->@inbounds(indices[j]!==i), 1:count_inner)
                indices[count+=1] = i
            end
        end
    end
    resize!(indices, count)
end
