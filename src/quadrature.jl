"""
    QuadratureRule(family, points, weights)

Points and weights for integration on a reference cell of `family`.
"""
struct QuadratureRule{F <: Shape, dim, T, P <: AbstractVector{Vec{dim, T}}, W <: AbstractVector{<: T}}
    points::P
    weights::W
end

function QuadratureRule(::Type{F}, points::P, weights::W) where {F <: Shape, dim, T, P <: AbstractVector{Vec{dim, T}}, W <: AbstractVector{<: T}}
    length(points) == length(weights) || throw(DimensionMismatch("quadrature points and weights must have the same length"))
    QuadratureRule{F, dim, T, P, W}(points, weights)
end

_reference_cell_family(::S) where {S <: Shape} = S
_reference_cell_family(::Line) = Line
_reference_cell_family(::Quad) = Quad
_reference_cell_family(::Hex) = Hex
_reference_cell_family(::Tri) = Tri
_reference_cell_family(::Tet) = Tet
_reference_cell_family(::QuadratureRule{F}) where {F} = F

_order_value(::Order{p}) where {p} = p

"""
    generate_quadrature_rule(shape::Shape)
    generate_quadrature_rule(T, shape::Shape)

Generate the standard [`QuadratureRule`](@ref) for `shape`. For interpolation
order `p`, the rule integrates degree `2p` in each reference coordinate for
tensor-product shapes and total degree `2p` for simplex shapes. `T` may be
`Float16`, `Float32`, or `Float64` and defaults to `Float64`. The reference-cell
family and interpolation order of `shape` determine the selected rule.
"""
generate_quadrature_rule(shape::Shape) = generate_quadrature_rule(Float64, shape)
function generate_quadrature_rule(::Type{T}, shape::Shape) where {T}
    generate_quadrature_rule(T, _reference_cell_family(shape), 2 * _order_value(get_order(shape)))
end

"""
    generate_quadrature_rule(family, exactness)
    generate_quadrature_rule(T, family, exactness)

Generate a rule for the FEM reference-cell `family` with at least the requested
polynomial `exactness`. For `Quad` and `Hex`, exactness applies in each
reference coordinate; for `Tri` and `Tet`, it applies to total degree.

`Line`, `Quad`, and `Hex` accept any nonnegative exactness, `Tri` supports
exactness from 0 through 6, and `Tet` from 0 through 7. `T` may be `Float16`,
`Float32`, or `Float64` and defaults to `Float64`.
"""
const ReferenceShapeFamily = Union{Line, Quad, Hex, Tri, Tet}

function _check_quadrature_float(::Type{T}) where {T}
    T in (Float16, Float32, Float64) || throw(ArgumentError("quadrature rule scalar type must be Float16, Float32, or Float64; got $T"))
    T
end

generate_quadrature_rule(family::Type{<: ReferenceShapeFamily}, exactness::Int) = generate_quadrature_rule(Float64, family, exactness)
generate_quadrature_rule(::Type{T}, ::Type{Line}, exactness::Int) where {T} = _generate_tensor_quadrature_rule(_check_quadrature_float(T), Line, exactness, Val(1))
generate_quadrature_rule(::Type{T}, ::Type{Quad}, exactness::Int) where {T} = _generate_tensor_quadrature_rule(_check_quadrature_float(T), Quad, exactness, Val(2))
generate_quadrature_rule(::Type{T}, ::Type{Hex}, exactness::Int) where {T} = _generate_tensor_quadrature_rule(_check_quadrature_float(T), Hex, exactness, Val(3))
generate_quadrature_rule(::Type{T}, ::Type{Tri}, exactness::Int) where {T} = _generate_simplex_quadrature_rule(_check_quadrature_float(T), Tri, exactness)
generate_quadrature_rule(::Type{T}, ::Type{Tet}, exactness::Int) where {T} = _generate_simplex_quadrature_rule(_check_quadrature_float(T), Tet, exactness)

function _check_quadrature_exactness(family, exactness::Int, maximum::Int)
    0 ≤ exactness ≤ maximum || throw(ArgumentError("$family quadrature supports exactness from 0 through $maximum; got $exactness"))
end

function _generate_tensor_quadrature_rule(::Type{T}, family, exactness::Int, ::Val{dim}) where {T, dim}
    exactness ≥ 0 || throw(ArgumentError("$family quadrature exactness must be nonnegative; got $exactness"))
    rule1d = _gauss_legendre_rule(T, Val(fld(exactness, 2) + 1))
    _tensor_product_quadrature_rule(family, ntuple(_ -> rule1d, Val(dim)))
end

_tensor_product_family(::Val{1}) = Line
_tensor_product_family(::Val{2}) = Quad
_tensor_product_family(::Val{3}) = Hex

function _tensor_product_quadrature_rule(family, rules::Tuple{Vararg{Tuple{<: StaticVector, <: StaticVector}, dim}}) where {dim}
    qpts1d = map(first, rules)
    qwts1d = map(last, rules)
    indices = CartesianIndices(map(length, qpts1d))
    npoints = length(indices)
    qpts = SVector{npoints}(map(I -> Vec(map(getindex, qpts1d, Tuple(I))), indices))
    qwts = SVector{npoints}(map(I -> prod(map(getindex, qwts1d, Tuple(I))), indices))
    QuadratureRule(family, qpts, qwts)
end

function _tensor_product_quadrature_rule(family, rules::Tuple{Vararg{Any, dim}}) where {dim}
    qpts1d = map(first, rules)
    qwts1d = map(last, rules)
    indices = CartesianIndices(map(length, qpts1d))
    qpts = vec(map(I -> Vec(map(getindex, qpts1d, Tuple(I))), indices))
    qwts = vec(map(I -> prod(map(getindex, qwts1d, Tuple(I))), indices))
    QuadratureRule(family, qpts, qwts)
end

_gauss_legendre_rule(::Type{T}, ::Degree{p}) where {T, p} = _gauss_legendre_rule(T, Val(p + 1))

function _gauss_legendre_rule(::Type{T}, ::Val{1}) where {T}
    SVector{1, T}((0,)), SVector{1, T}((2,))
end
function _gauss_legendre_rule(::Type{T}, ::Val{2}) where {T}
    x = T(0.57735026918962576451)
    SVector{2, T}((-x, x)), SVector{2, T}((1, 1))
end
function _gauss_legendre_rule(::Type{T}, ::Val{3}) where {T}
    x = T(0.77459666924148337704)
    w1 = T(0.55555555555555555556)
    w2 = T(0.88888888888888888889)
    SVector{3, T}((-x, 0, x)), SVector{3, T}((w1, w2, w1))
end
function _gauss_legendre_rule(::Type{T}, ::Val{4}) where {T}
    x1 = T(0.33998104358485626480)
    x2 = T(0.86113631159405257522)
    w1 = T(0.65214515486254614263)
    w2 = T(0.34785484513745385737)
    SVector{4, T}((-x2, -x1, x1, x2)), SVector{4, T}((w2, w1, w1, w2))
end
function _gauss_legendre_rule(::Type{T}, ::Val{5}) where {T}
    x1 = T(0.53846931010568309104)
    x2 = T(0.90617984593866399280)
    w1 = T(0.47862867049936646804)
    w2 = T(0.23692688505618908751)
    w3 = T(0.56888888888888888889)
    SVector{5, T}((-x2, -x1, 0, x1, x2)), SVector{5, T}((w2, w1, w3, w1, w2))
end
function _gauss_legendre_rule(::Type{T}, ::Val{6}) where {T}
    x1 = T(0.23861918608319690863)
    x2 = T(0.66120938646626451366)
    x3 = T(0.93246951420315202781)
    w1 = T(0.46791393457269104739)
    w2 = T(0.36076157304813860757)
    w3 = T(0.17132449237917034504)
    SVector{6, T}((-x3, -x2, -x1, x1, x2, x3)), SVector{6, T}((w3, w2, w1, w1, w2, w3))
end

function _gauss_legendre_rule(::Type{T}, ::Val{n}) where {T, n}
    n ≥ 1 || throw(ArgumentError("Gauss–Legendre quadrature requires at least one point; got $n"))
    β = [i / sqrt(4i^2 - 1) for i in 1:n-1]
    decomposition = eigen(SymTridiagonal(zeros(n), β))
    T.(decomposition.values), T.(2 .* abs2.(decomposition.vectors[1,:]))
end

# Positive symmetric simplex rules of Witherden and Vincent (2015).
_simplex_rule_degrees(::Type{Tri}) = (1, 1, 2, 4, 4, 6, 6)
_simplex_rule_degrees(::Type{Tet}) = (1, 1, 2, 5, 5, 5, 7, 7)

function _generate_simplex_quadrature_rule(::Type{T}, family, exactness::Int) where {T}
    rule_degrees = _simplex_rule_degrees(family)
    _check_quadrature_exactness(family, exactness, length(rule_degrees) - 1)
    _simplex_quadrature_rule(T, family, Val(rule_degrees[exactness + 1]))
end

@inline _triangle_orbit3(a::T, b::T) where {T} = SVector{3, Vec{2, T}}(((a, a), (a, b), (b, a)))
@inline _triangle_orbit6(a::T, b::T, c::T) where {T} = SVector{6, Vec{2, T}}(((a, b), (b, a), (a, c), (c, a), (b, c), (c, b)))

function _simplex_quadrature_rule(::Type{T}, ::Type{Tri}, ::Val{1}) where {T}
    a = T(0.33333333333333333333)
    w = T(0.50000000000000000000)
    QuadratureRule(Tri, SVector{1, Vec{2, T}}(((a, a),)), SVector{1, T}((w,)))
end

function _simplex_quadrature_rule(::Type{T}, ::Type{Tri}, ::Val{2}) where {T}
    a = T(0.16666666666666666667)
    b = T(0.66666666666666666667)
    w = T(0.16666666666666666667)
    QuadratureRule(Tri, _triangle_orbit3(a, b), SVector{3, T}((w, w, w)))
end

function _simplex_quadrature_rule(::Type{T}, ::Type{Tri}, ::Val{4}) where {T}
    a1 = T(0.44594849091596488632)
    b1 = T(0.10810301816807022736)
    w1 = T(0.11169079483900573285)
    a2 = T(0.09157621350977074346)
    b2 = T(0.81684757298045851308)
    w2 = T(0.05497587182766093382)
    points = SVector{6, Vec{2, T}}((_triangle_orbit3(a1, b1)..., _triangle_orbit3(a2, b2)...))
    QuadratureRule(Tri, points, SVector{6, T}((w1, w1, w1, w2, w2, w2)))
end

function _simplex_quadrature_rule(::Type{T}, ::Type{Tri}, ::Val{6}) where {T}
    a1 = T(0.06308901449150222662)
    b1 = T(0.87382197101699554676)
    w1 = T(0.02542245318510340940)
    a2 = T(0.24928674517091042873)
    b2 = T(0.50142650965817914255)
    w2 = T(0.05839313786318968413)
    a3 = T(0.63650249912139866826)
    b3 = T(0.31035245103378439335)
    c3 = T(0.05314504984481693839)
    w3 = T(0.04142553780918678541)
    points = SVector{12, Vec{2, T}}((_triangle_orbit3(a1, b1)..., _triangle_orbit3(a2, b2)..., _triangle_orbit6(a3, b3, c3)...))
    weights = SVector{12, T}((w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3))
    QuadratureRule(Tri, points, weights)
end

@inline _tetrahedron_orbit4(a::T, b::T) where {T} = SVector{4, Vec{3, T}}(((a, b, b), (b, a, b), (b, b, a), (b, b, b)))
@inline _tetrahedron_orbit6(a::T, b::T) where {T} = SVector{6, Vec{3, T}}(((b, b, a), (b, a, a), (a, a, b), (a, b, a), (b, a, b), (a, b, b)))
@inline _tetrahedron_orbit12(a::T, b::T, c::T) where {T} = SVector{12, Vec{3, T}}(((a, a, b), (a, b, a), (b, a, a), (a, a, c), (a, c, a), (c, a, a), (a, b, c), (a, c, b), (b, c, a), (b, a, c), (c, a, b), (c, b, a)))

function _simplex_quadrature_rule(::Type{T}, ::Type{Tet}, ::Val{1}) where {T}
    a = T(0.25000000000000000000)
    w = T(0.16666666666666666667)
    QuadratureRule(Tet, SVector{1, Vec{3, T}}(((a, a, a),)), SVector{1, T}((w,)))
end

function _simplex_quadrature_rule(::Type{T}, ::Type{Tet}, ::Val{2}) where {T}
    a = T(0.58541019662496845446)
    b = T(0.13819660112501051518)
    w = T(0.04166666666666666667)
    QuadratureRule(Tet, _tetrahedron_orbit4(a, b), SVector{4, T}((w, w, w, w)))
end

function _simplex_quadrature_rule(::Type{T}, ::Type{Tet}, ::Val{5}) where {T}
    a1 = T(0.31088591926330060980)
    b1 = T(0.06734224221009817060)
    w1 = T(0.01878132095300264180)
    a2 = T(0.09273525031089122640)
    b2 = T(0.72179424906732632079)
    w2 = T(0.01224884051939365826)
    a3 = T(0.04550370412564964949)
    b3 = T(0.45449629587435035051)
    w3 = T(0.00709100346284691107)
    points = SVector{14, Vec{3, T}}((_tetrahedron_orbit4(b1, a1)..., _tetrahedron_orbit4(b2, a2)..., _tetrahedron_orbit6(a3, b3)...))
    weights = SVector{14, T}((w1, w1, w1, w1, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3))
    QuadratureRule(Tet, points, weights)
end

function _simplex_quadrature_rule(::Type{T}, ::Type{Tet}, ::Val{7}) where {T}
    q = T(0.25000000000000000000)
    w0 = T(0.01591421491068847546)
    a1 = T(0.31570114977820279423)
    b1 = T(0.05289655066539161732)
    w1 = T(0.00705493020166117132)
    a2 = T(0.44951017740160364999)
    b2 = T(0.05048982259839635001)
    w2 = T(0.00531615463880959638)
    a3 = T(0.18883383102600115322)
    b3 = T(0.57517163758699996201)
    c3 = T(0.04716070036099773155)
    w3 = T(0.00620118845472243663)
    a4 = T(0.02126547254148325461)
    b4 = T(0.81083024109854862083)
    c4 = T(0.14663881381848486996)
    w4 = T(0.00135179513831722360)
    points = SVector{35, Vec{3, T}}(((q, q, q), _tetrahedron_orbit4(b1, a1)..., _tetrahedron_orbit6(a2, b2)..., _tetrahedron_orbit12(a3, b3, c3)..., _tetrahedron_orbit12(a4, b4, c4)...))
    weights = SVector{35, T}((w0, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2, w3, w3, w3, w3, w3, w3, w3, w3, w3, w3, w3, w3, w4, w4, w4, w4, w4, w4, w4, w4, w4, w4, w4, w4))
    QuadratureRule(Tet, points, weights)
end
