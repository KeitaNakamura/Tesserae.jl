using ..Tesserae: get_device, CPUDevice, GPUDevice

shift(d::Int, s::Int, ::Val{dim}) where {dim} = CartesianIndex(ntuple(i -> i==d ? s : 0, Val(dim)))
shift1(d::Int, ::Val{dim}) where {dim} = shift(d, 1, Val(dim))

function shift_radius(offsets::AbstractArray{CartesianIndex{dim}}) where {dim}
    ntuple(d -> maximum(abs(off[d]) for off in offsets), Val(dim))
end

@generated function dot_unrolled(dest, src, offsets::SVector{N}, weights::SVector{N}, baseshift, I) where {N}
    ex = :(weights[1] * src[I + baseshift + offsets[1]])
    for i in 2:N
        ex = :(muladd(weights[$i], src[I + baseshift + offsets[$i]], $ex))
    end
    quote
        @inline
        @inbounds $ex
    end
end

# CPU
function _stencil!(::CPUDevice, combine, dest, src, offsets, weights, baseshift, indices)
    @inbounds @simd for I in indices
        tmp = dot_unrolled(dest, src, offsets, weights, baseshift, I)
        dest[I] = combine(dest[I], tmp)
    end
end

# GPU
@kernel function kernel_stencil(combine, dest, @Const(src), @Const(offsets), @Const(weights), @Const(baseshift), @Const(I0))
    I = I0 - oneunit(I0) + @index(Global, Cartesian)
    tmp = zero(promote_type(eltype(src), eltype(weights)))
    @inbounds begin
        tmp = dot_unrolled(dest, src, offsets, weights, baseshift, I)
        dest[I] = combine(dest[I], tmp)
    end
end
function _stencil!(::GPUDevice, combine, dest, src, offsets, weights, baseshift, indices::CartesianIndices)
    backend = get_backend(dest)
    kernel = kernel_stencil(backend)
    kernel(combine, dest, src, offsets, weights, baseshift, first(indices); ndrange=size(indices))
    synchronize(backend)
end

function stencil!(combine, dest::StencilArray, src::StencilArray, offsets::AbstractArray{CartesianIndex{dim}}, weights::AbstractArray{<: Number}; pad::Int) where {dim}
    @assert get_device(dest) == get_device(src)
    @assert ndims(dest) == ndims(src) == dim
    @assert length(offsets) == length(weights)
    check_size(dest, src)

    rad = shift_radius(offsets)
    all(d -> pad ≥ rad[d], 1:dim) || throw(ArgumentError("pad=$pad is too small for this stencil; required ≥ $(maximum(rad))"))

    ranges = ntuple(Val(dim)) do d
        lo = firstindex(dest, d) + pad
        hi = lastindex(dest, d) - pad
        (lo ≤ hi) || throw(ArgumentError("pad=$pad is too large for dest (axis=$d)"))
        lo:hi
    end

    baseshift = CartesianIndex(ntuple(d -> firstindex(src, d) - firstindex(dest, d), Val(dim)))
    _stencil!(get_device(dest), combine, dest, src, offsets, weights, baseshift, CartesianIndices(ranges))
    return dest
end

@inline _replace(old, new) = new
function stencil!(dest::StencilArray, src::StencilArray, offsets::AbstractArray{CartesianIndex{dim}}, weights::AbstractArray{<: Number}; pad::Int) where {dim}
    stencil!(_replace, dest, src, offsets, weights; pad)
end

function stencil_sparse(destloc::Location, src::StencilArray, offsets::AbstractVector{CartesianIndex{dim}}, weights::AbstractVector{<:Number}; pad::Int) where {dim}
    rad = shift_radius(offsets)
    all(d -> pad ≥ rad[d], 1:dim) || throw(ArgumentError("pad=$pad is too small for this stencil; required ≥ $(maximum(rad))"))

    srcdims = size(src)
    destdims = infersize(destloc, getlocation(src), srcdims)

    # assemble only interior as `stencil!`
    interior = interior_indices(destloc, Base.OneTo.(destdims); pad)

    nrows = prod(destdims)
    ncols = prod(srcdims)

    LIdest = LinearIndices(destdims)
    LIsrc  = LinearIndices(srcdims)

    T = promote_type(eltype(src), eltype(weights))
    nnz = length(interior) * length(weights)

    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{T}(undef, nnz)

    count = 1
    @inbounds for i in interior
        row = LIdest[i]
        for j in eachindex(weights, offsets)
            I[count] = row
            J[count] = LIsrc[i + offsets[j]]
            V[count] = weights[j]
            count += 1
        end
    end

    return sparse(I, J, V, nrows, ncols)
end
