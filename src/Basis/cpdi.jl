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
struct CPDI <: Basis end

cpdi_spgrid_error(context) = error("$context: CPDI is currently supported only on dense Grid, not SpGrid")
@inline supportnodes(::BasisWeight{CPDI}, ::SpGrid) = cpdi_spgrid_error("supportnodes")

struct CPDISupportNodes{T, V <: AbstractVector{T}} <: AbstractVector{T}
    indices::V
    len::Int
end
Base.size(x::CPDISupportNodes) = (x.len,)
@inline function Base.getindex(x::CPDISupportNodes, i::Int)
    @boundscheck checkbounds(x, i)
    @inbounds x.indices[i]
end

@generated function allocate_basis_values(::Type{Vec{dim, T}}, ::CPDI; derivative::Order{k}, name::Val{sym}=Val(:w)) where {dim, T, k, sym}
    k == 1 || return :(throw(ArgumentError("CPDI supports derivative=Order(1) only")))
    w = sym
    ∇w = Symbol(:∇, sym)
    quote
        len = 2^dim * 2^dim # maximum length
        (; $w=fill(zero(T), len), $∇w=fill(zero(Vec{dim, T}), len))
    end
end

function initial_supportnodes(::CPDI, mesh::CartesianMesh{dim}) where {dim}
    maxlength = 2^dim * 2^dim
    indices = zero(SVector{maxlength, CartesianIndex{dim}})
    CPDISupportNodes(indices, 0)
end

function update_basis_weight!(bw::BasisWeight{CPDI}, pt, mesh::CartesianMesh{1})
    xₚ = getx(pt)
    r₁ = pt.F * Vec(pt.l/2)
    x₁ = xₚ - r₁
    x₂ = xₚ + r₁
    Vₚ = 2r₁[1]

    indices = collect_cpdi_supportnodes(Val(4), (x₁, x₂), mesh)
    supportnodes_storage(bw)[] = indices

    spline = BSpline(Linear())
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        w₁ = only(nodal_basis_jet(Order(0), spline, x₁, mesh, i))
        w₂ = only(nodal_basis_jet(Order(0), spline, x₂, mesh, i))
        w = (w₁ + w₂) / 2
        ∇w = Vec(w₂ - w₁) / Vₚ
        set_values!(bw, ip, (w,∇w))
    end
end

function update_basis_weight!(bw::BasisWeight{CPDI}, pt, mesh::CartesianMesh{2})
    xₚ = getx(pt)
    r₁ = pt.F * Vec(pt.l/2,0)
    r₂ = pt.F * Vec(0,pt.l/2)
    x₁ = xₚ - r₁ - r₂
    x₂ = xₚ + r₁ - r₂
    x₃ = xₚ + r₁ + r₂
    x₄ = xₚ - r₁ + r₂
    Vₚ = 4 * abs(r₁ × r₂)

    indices = collect_cpdi_supportnodes(Val(16), (x₁, x₂, x₃, x₄), mesh)
    supportnodes_storage(bw)[] = indices

    a = Vec(r₁[2]-r₂[2], r₂[1]-r₁[1])
    b = Vec(r₁[2]+r₂[2],-r₁[1]-r₂[1])
    spline = BSpline(Linear())
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        w₁ = only(nodal_basis_jet(Order(0), spline, x₁, mesh, i))
        w₂ = only(nodal_basis_jet(Order(0), spline, x₂, mesh, i))
        w₃ = only(nodal_basis_jet(Order(0), spline, x₃, mesh, i))
        w₄ = only(nodal_basis_jet(Order(0), spline, x₄, mesh, i))
        w = (w₁ + w₂ + w₃ + w₄) / 4
        ∇w = ((w₁-w₃)*a + (w₂-w₄)*b) / Vₚ
        set_values!(bw, ip, (w,∇w))
    end
end

function update_basis_weight!(bw::BasisWeight{CPDI}, pt, mesh::CartesianMesh{3})
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

    indices = collect_cpdi_supportnodes(Val(64), (x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈), mesh)
    supportnodes_storage(bw)[] = indices

    A = hcat(r₂×r₃, r₃×r₁, r₁×r₂) / Vₚ
    spline = BSpline(Linear())
    @inbounds for ip in eachindex(indices)
        i = indices[ip]
        w₁ = only(nodal_basis_jet(Order(0), spline, x₁, mesh, i))
        w₂ = only(nodal_basis_jet(Order(0), spline, x₂, mesh, i))
        w₃ = only(nodal_basis_jet(Order(0), spline, x₃, mesh, i))
        w₄ = only(nodal_basis_jet(Order(0), spline, x₄, mesh, i))
        w₅ = only(nodal_basis_jet(Order(0), spline, x₅, mesh, i))
        w₆ = only(nodal_basis_jet(Order(0), spline, x₆, mesh, i))
        w₇ = only(nodal_basis_jet(Order(0), spline, x₇, mesh, i))
        w₈ = only(nodal_basis_jet(Order(0), spline, x₈, mesh, i))
        w = (w₁ + w₂ + w₃ + w₄ + w₅ + w₆ + w₇ + w₈) / 8
        ∇w = A * Vec(-w₁+w₂+w₃-w₄-w₅+w₆+w₇-w₈, -w₁-w₂+w₃+w₄-w₅-w₆+w₇+w₈, -w₁-w₂-w₃-w₄+w₅+w₆+w₇+w₈)
        set_values!(bw, ip, (w,∇w))
    end
end

@inline function collect_cpdi_supportnodes(::Val{L}, corners, mesh::CartesianMesh{dim}) where {L, dim}
    indices = MVector{L, CartesianIndex{dim}}(undef) # this works on GPU (https://discourse.julialang.org/t/cudanative-dynamic-allocation/35435/5)
    count = 0
    @inbounds for x in corners
        count_inner = count
        for i in supportnodes(BSpline(Linear()), x, mesh)
            if all(j->@inbounds(indices[j]!==i), 1:count_inner)
                indices[count+=1] = i
            end
        end
    end
    CPDISupportNodes(SVector(indices), count)
end
