struct Diff{order, radius}
    axis::Int
    function Diff{order, radius}(axis::Int) where {order, radius}
        new{order::Int, radius::Int}(axis)
    end
end
Diff(axis::Int) = Diff{1,1}(axis)
Diff{order}(axis::Int) where {order} = Diff{order,1}(axis)
getaxis(x::Diff) = x.axis

# generate_offsets
generate_offsets(op::Diff, dest::Cell, src::Cell) = _generate_offsets(op, dest, src)
generate_offsets(op::Diff, dest::Cell, src::Face) = _generate_offsets(op, dest, src)
generate_offsets(op::Diff, dest::Face, src::Cell) = (@assert getaxis(op) == getaxis(dest); _generate_offsets(op, dest, src))
generate_offsets(op::Diff, dest::Face, src::Face) = (@assert getaxis(dest) == getaxis(src); _generate_offsets(op, dest, src))
_generate_offsets(::Diff{1, 1}, ::Cell, ::Cell) = (-1, 1)
_generate_offsets(::Diff{1, 2}, ::Cell, ::Cell) = (-2, -1, 1, 2)
_generate_offsets(::Diff{1, 3}, ::Cell, ::Cell) = (-3, -2, -1, 1, 2, 3)
_generate_offsets(::Diff{1, 1}, ::Face, ::Face) = (-1, 1)
_generate_offsets(::Diff{1, 2}, ::Face, ::Face) = (-2, -1, 1, 2)
_generate_offsets(::Diff{1, 3}, ::Face, ::Face) = (-3, -2, -1, 1, 2, 3)
_generate_offsets(::Diff{1, 1}, ::Face, ::Cell) = (-1, 0)
_generate_offsets(::Diff{1, 2}, ::Face, ::Cell) = (-2, -1, 0, 1)
_generate_offsets(::Diff{1, 3}, ::Face, ::Cell) = (-3, -2, -1, 0, 1, 2)
_generate_offsets(::Diff{1, 1}, ::Cell, ::Face) = (0, 1)
_generate_offsets(::Diff{1, 2}, ::Cell, ::Face) = (-1, 0, 1, 2)
_generate_offsets(::Diff{1, 3}, ::Cell, ::Face) = (-2, -1, 0, 1, 2, 3)
_generate_offsets(::Diff{2, 1}, ::Cell, ::Cell) = (-1, 0, 1)
_generate_offsets(::Diff{2, 2}, ::Cell, ::Cell) = (-2, -1, 0, 1, 2)
_generate_offsets(::Diff{2, 3}, ::Cell, ::Cell) = (-3, -2, -1, 0, 1, 2, 3)
_generate_offsets(::Diff{2, 1}, ::Face, ::Face) = (-1, 0, 1)
_generate_offsets(::Diff{2, 2}, ::Face, ::Face) = (-2, -1, 0, 1, 2)
_generate_offsets(::Diff{2, 3}, ::Face, ::Face) = (-3, -2, -1, 0, 1, 2, 3)

# generate_weights
generate_weights(op::Diff, dest::Cell, src::Cell) = _generate_weights(op, dest, src)
generate_weights(op::Diff, dest::Cell, src::Face) = _generate_weights(op, dest, src)
generate_weights(op::Diff, dest::Face, src::Cell) = (@assert getaxis(op) == getaxis(dest); _generate_weights(op, dest, src))
generate_weights(op::Diff, dest::Face, src::Face) = (@assert getaxis(dest) == getaxis(src); _generate_weights(op, dest, src))
_generate_weights(::Diff{1, 1}, ::Cell, ::Cell) = (-1/2, 1/2)
_generate_weights(::Diff{1, 2}, ::Cell, ::Cell) = ( 1/12, -2/3, 2/3, -1/12)
_generate_weights(::Diff{1, 3}, ::Cell, ::Cell) = (-1/60, 3/20, -3/4, 3/4, -3/20, 1/60)
_generate_weights(::Diff{1, 1}, ::Face, ::Face) = (-1/2, 1/2)
_generate_weights(::Diff{1, 2}, ::Face, ::Face) = ( 1/12, -2/3, 2/3, -1/12)
_generate_weights(::Diff{1, 3}, ::Face, ::Face) = (-1/60, 3/20, -3/4, 3/4, -3/20, 1/60)
_generate_weights(::Diff{1, 1}, ::Face, ::Cell) = (-1.0, 1.0)
_generate_weights(::Diff{1, 2}, ::Face, ::Cell) = ( 1/24, -9/8, 9/8, -1/24)
_generate_weights(::Diff{1, 3}, ::Face, ::Cell) = (-3/640, 25/384, -75/64, 75/64, -25/384, 3/640)
_generate_weights(::Diff{1, 1}, ::Cell, ::Face) = (-1.0, 1.0)
_generate_weights(::Diff{1, 2}, ::Cell, ::Face) = ( 1/24, -9/8, 9/8, -1/24)
_generate_weights(::Diff{1, 3}, ::Cell, ::Face) = (-3/640, 25/384, -75/64, 75/64, -25/384, 3/640)
_generate_weights(::Diff{2, 1}, ::Cell, ::Cell) = ( 1.0, -2.0, 1.0)
_generate_weights(::Diff{2, 2}, ::Cell, ::Cell) = (-1/12, 4/3, -5/2, 4/3, -1/12)
_generate_weights(::Diff{2, 3}, ::Cell, ::Cell) = ( 1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90)
_generate_weights(::Diff{2, 1}, ::Face, ::Face) = ( 1.0, -2.0, 1.0)
_generate_weights(::Diff{2, 2}, ::Face, ::Face) = (-1/12, 4/3, -5/2, 4/3, -1/12)
_generate_weights(::Diff{2, 3}, ::Face, ::Face) = ( 1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90)

#=
@generated function _reduce_stencil_coeffs(::Val{offsets}, ::Val{weights}) where {offsets, weights}
    offsets_tmp = similar(offsets)
    weights_tmp = similar(weights)
    offsets_tmp[1] = offsets[1]
    weights_tmp[1] = weights[1]
    count = 1
    for j in 2:length(offsets)
        if offsets[j] === offsets[count]
            weights_tmp[count] += weights[j]
        else
            count += 1
            offsets_tmp[count] = offsets[j]
            weights_tmp[count] = weights[j]
        end
    end
    quote
        SVector($(offsets_tmp[1:count]...)), SVector($(weights_tmp[1:count]...))
    end
end

function reduce_stencil_coeffs(offsets::SVector{N, CartesianIndex{dim}}, weights::SVector{N, T}) where {N, dim, T}
    # _reduce_stencil_coeffs(Val(offsets), Val(weights))
    offsets_tmp = similar(offsets)
    weights_tmp = similar(weights)
    offsets_tmp[1] = offsets[1]
    weights_tmp[1] = weights[1]
    count = 1
    for j in 2:length(offsets)
        if offsets[j] === offsets[count]
            weights_tmp[count] += weights[j]
        else
            count += 1
            offsets_tmp[count] = offsets[j]
            weights_tmp[count] = weights[j]
        end
    end
    SVector(offsets_tmp[SOneTo(count)]), SVector(weights_tmp[SOneTo(count)]...)
end

function reorder_stencil_coeffs(offsets::SVector{N, CartesianIndex{dim}}, weights::SVector{N}) where {N, dim}
    mins = ntuple(d -> minimum(off -> off[d], offsets), Val(dim))
    maxs = ntuple(d -> maximum(off -> off[d], offsets), Val(dim))
    dims = ntuple(d -> maxs[d] - mins[d] + 1, Val(dim))
    LI = LinearIndices(dims)
    key(off) = LI[CartesianIndex(ntuple(d -> off[d] - mins[d] + 1, Val(dim)))]
    p = sortperm(offsets; by = key)
    offsets[p], weights[p]
end
=#

function stencil_coeffs(op::Diff{order}, dest::Location, src::Location, h::NTuple{dim, T}) where {order, dim, T <: Real}
    ax = getaxis(op)
    (1 ≤ ax ≤ dim) || error("axis must be in 1:$dim, got $ax")
    Tf = float(T)
    offsets = map(offset -> shift(ax, offset, Val(dim)), generate_offsets(op, dest, src))
    weights = map(w -> Tf(w) / h[ax]^order, generate_weights(op, dest, src))
    return SVector(offsets), SVector(weights)
end

function stencil!(combine, diff::Diff, dest::StencilArray, src::StencilArray; pad::Int, spacing::Real)
    @assert ndims(dest) == ndims(src)
    h = ntuple(_ -> spacing, Val(ndims(dest)))
    offsets, weights = stencil_coeffs(diff, getlocation(dest), getlocation(src), h)
    stencil!(combine, dest, src, offsets, weights; pad)
end
function stencil!(diff::Diff, dest::StencilArray, src::StencilArray; pad::Int, spacing::Real)
    stencil!(_replace, diff, dest, src; pad, spacing)
end

abstract type Operator end

struct Gradient{r} <: Operator end
struct Divergence{r} <: Operator end
struct Laplacian{r} <: Operator end
struct Curl{r} <: Operator end

Gradient() = Gradient{1}()
Divergence() = Divergence{1}()
Laplacian() = Laplacian{1}()
Curl() = Curl{1}()

# Gradient: Cell → Face
function stencil!(::Gradient{r}, dests::NTuple{dim, <: StencilArray{Face, T, dim}}, src::StencilArray{Cell, T, dim}; pad::Int, spacing::Real) where {r, dim, T <: Real}
    for d in 1:dim
        stencil!(Diff{1,r}(d), dests[d], src; pad, spacing)
    end
    return dests
end
function stencil(op::Gradient, src::StencilArray{Cell, T, dim}; pad::Int, spacing::Real) where {dim, T <: Real}
    dests = ntuple(d -> similar(src, Face(d)), Val(dim))
    stencil!(op, dests, src; pad, spacing)
end

# Gradient: Face → Cell
function stencil!(::Gradient{r}, dest::StencilArray{Cell, <: Any, dim}, srcs::NTuple{dim, <: StencilArray{Face, <: Any, dim}}; pad::Int, spacing::Real) where {r, dim}
    fillzero!(inner(dest; pad))
    for i in 1:dim
        src = srcs[i]
        eᵢ = Vec{dim}(==(i))
        for j in 1:dim
            eⱼ = Vec{dim}(==(j))
            eiej = eᵢ ⊗ eⱼ
            stencil!((old,new)->old+new⊗eᵢeⱼ, Diff{1,r}(j), dest, src; pad, spacing)
        end
    end
    return dest
end

# Divergence: Face → Cell
function stencil!(::Divergence{r}, dest::StencilArray{Cell, T, dim}, srcs::NTuple{dim, <: StencilArray{Face, T, dim}}; pad::Int, spacing::Real) where {r, dim, T <: Real}
    inner(dest; pad) .= zero(T)
    for d in 1:dim
        stencil!(+, Diff{1,r}(d), dest, srcs[d]; pad, spacing)
    end
    return dest
end
function stencil(op::Divergence, srcs::NTuple{dim, <: StencilArray{Face, T, dim}}; pad::Int, spacing::Real) where {dim, T <: Real}
    @assert allequal(infersize(Cell(), getlocation(srcs[d]), size(srcs[d])) for d in 1:dim)
    dest = similar(srcs[1], Cell())
    stencil!(op, dest, srcs; pad, spacing)
end

# Laplacian: Cell → Cell / Face → Face
function stencil!(::Laplacian{r}, dest::StencilArray{Loc, T, dim}, src::StencilArray{Loc, T, dim}; pad::Int, spacing::Real) where {r, dim, Loc <: Union{Cell, Face}, T <: Real}
    destloc = getlocation(dest)
    srcloc = getlocation(src)
    h = ntuple(_ -> spacing, Val(dim))
    offsets, weights = reduce(ntuple(d -> stencil_coeffs(Diff{2,r}(d), destloc, srcloc, h), Val(dim))) do left, right
        vcat(left[1], right[1]), vcat(left[2], right[2])
    end
    stencil!(dest, src, offsets, weights; pad)
    dest
end
function stencil(op::Laplacian, src::StencilArray{Loc, T}; pad::Int, spacing::Real) where {Loc <: Union{Cell, Face}, T <: Real}
    dest = similar(src)
    stencil!(op, dest, src; pad, spacing)
end

# Curl: Face → Cell
function stencil!(::Curl{r}, dest::StencilArray{Cell, T, 2}, srcs::NTuple{2, <: StencilArray{Face, T, 2}}; pad::Int, spacing::Real) where {r, T}
    # ∂v/∂x - ∂u/∂y
    stencil!(Diff{1,r}(1), dest, srcs[2]; pad, spacing)
    stencil!(-, Diff{1,r}(2), dest, srcs[1]; pad, spacing)
    return dest
end
function stencil!(::Curl{r}, dest::StencilArray{Cell, Vec{3, T}, 3}, srcs::NTuple{3, <: StencilArray{Face, T, 3}}; pad::Int, spacing::Real) where {r, T}
    inner(dest; pad) .= Ref(zero(Vec{3, T}))

    # ωx = ∂w/∂y - ∂v/∂z
    e = Vec{3,T}(==(1))
    stencil!((old,new)->old+new*e, Diff{1,r}(2), dest, srcs[3]; pad, spacing)
    stencil!((old,new)->old-new*e, Diff{1,r}(3), dest, srcs[2]; pad, spacing)

    # ωy = ∂u/∂z - ∂w/∂x
    e = Vec{3,T}(==(2))
    stencil!((old,new)->old+new*e, Diff{1,r}(3), dest, srcs[1]; pad, spacing)
    stencil!((old,new)->old-new*e, Diff{1,r}(1), dest, srcs[3]; pad, spacing)

    # ωz = ∂v/∂x - ∂u/∂y
    e = Vec{3,T}(==(3))
    stencil!((old,new)->old+new*e, Diff{1,r}(1), dest, srcs[2]; pad, spacing)
    stencil!((old,new)->old-new*e, Diff{1,r}(2), dest, srcs[1]; pad, spacing)

    return dest
end
function stencil(op::Curl, srcs::NTuple{2, <: StencilArray{Face, T, 2}}; pad::Int, spacing::Real) where {T <: Real}
    @assert allequal(infersize(Cell(), getlocation(srcs[d]), size(srcs[d])) for d in 1:2)
    dest = similar(srcs[1], T, Cell())
    stencil!(op, dest, srcs; pad, spacing)
end
function stencil(op::Curl, srcs::NTuple{3, <: StencilArray{Face, T, 3}}; pad::Int, spacing::Real) where {T <: Real}
    @assert allequal(infersize(Cell(), getlocation(srcs[d]), size(srcs[d])) for d in 1:3)
    dest = similar(srcs[1], Vec{3, T}, Cell())
    stencil!(op, dest, srcs; pad, spacing)
end
