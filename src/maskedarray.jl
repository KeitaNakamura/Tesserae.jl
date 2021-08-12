struct Mask{dim} <: AbstractArray{Bool, dim}
    indices::Array{Int, dim}
end

Mask(dims::Tuple{Vararg{Int}}) = Mask(fill(-1, dims))
Mask(dims::Int...) = Mask(dims)

Base.size(mask::Mask) = size(mask.indices)
Base.IndexStyle(::Type{<: Mask}) = IndexLinear()

@inline Base.getindex(mask::Mask, i::Int) = (@_propagate_inbounds_meta; mask.indices[i] !== -1)
@inline Base.setindex!(mask::Mask, v, i::Int) = (@_propagate_inbounds_meta; mask.indices[i] = ifelse(convert(Bool, v), 0, -1))

Base.fill!(mask::Mask, v) = (fill!(mask.indices, ifelse(convert(Bool, v), 0, -1)); mask)

function reinit!(mask::Mask)
    count = 0
    for i in eachindex(mask)
        @inbounds mask.indices[i] = (mask[i] ? count += 1 : -1)
    end
    count
end

Base.copy(mask::Mask) = Mask(copy(mask.indices))

Broadcast.BroadcastStyle(::Type{<: Mask}) = ArrayStyle{Mask}()
Base.similar(bc::Broadcasted{ArrayStyle{Mask}}, ::Type{Bool}) = Mask(size(bc))


struct MaskedArray{T, dim, V <: AbstractVector{T}} <: AbstractArray{T, dim}
    data::V
    mask::Mask{dim}
end

function MaskedArray{T}(u::UndefInitializer, dims::Tuple{Vararg{Int}}) where {T}
    data = Vector{T}(u, prod(dims))
    mask = Mask(dims)
    MaskedArray(data, mask)
end
MaskedArray{T}(u::UndefInitializer, dims::Int...) where {T} = MaskedArray{T}(u, dims)

Base.IndexStyle(::Type{<: MaskedArray}) = IndexLinear()
Base.size(x::MaskedArray) = size(x.mask)

Base.propertynames(x::MaskedArray) = (:mask, :data, propertynames(x.data)...)
function Base.getproperty(x::MaskedArray, name::Symbol)
    name == :mask && return getfield(x, :mask)
    name == :data && return getfield(x, :data)
    MaskedArray(getproperty(getfield(x, :data), name), getfield(x, :mask))
end

@inline function Base.getindex(x::MaskedArray, i::Int)
    @boundscheck checkbounds(x, i)
    mask = x.mask
    @inbounds mask[i] ? x.data[mask.indices[i]] : initval(eltype(x))
end
@inline function Base.setindex!(x::MaskedArray, v, i::Int)
    @boundscheck checkbounds(x, i)
    mask = x.mask
    @inbounds begin
        mask[i] || throw(UndefRefError())
        x.data[mask.indices[i]] = v
    end
    x
end

function reinit!(x::MaskedArray{T}) where {T}
    n = reinit!(x.mask)
    resize!(x.data, n)
    reinit!(x.data)
    x
end

Base.unaliascopy(x::MaskedArray) = MaskedArray(Base.unaliascopy(x.data), Base.copy(x.mask))


Broadcast.BroadcastStyle(::Type{<: MaskedArray}) = ArrayStyle{MaskedArray}()

__extract_masks(masks::Tuple, x::Any) = masks
__extract_masks(masks::Tuple, x::AbstractArray) = (masks..., nothing)
_extract_masks(masks::Tuple, x::AbstractArray) = __extract_masks(masks, broadcastable(x)) # handle Tensor
_extract_masks(masks::Tuple, x::MaskedArray) = (masks..., x.mask)
_extract_masks(masks::Tuple, x::Any) = masks
extract_masks(masks::Tuple, args::Tuple{}) = masks
extract_masks(masks::Tuple, args::Tuple) = extract_masks(_extract_masks(masks, args[1]), Base.tail(args))
identical_mask(args...) = (masks = extract_masks((), args); all(x -> x === masks[1], masks))

getdata(x::MaskedArray) = x.data
getdata(x::Any) = x

getmask(x::MaskedArray) = x.mask
getmask(x::AbstractArray) = ifelse(broadcastable(x) isa AbstractArray, true, false) # handle Tensor
getmask(x::Any) = false

function Base.similar(bc::Broadcasted{ArrayStyle{MaskedArray}}, ::Type{ElType}) where {ElType}
    mask = broadcast(|, getmask.(bc.args)...)
    reinit!(MaskedArray(Vector{ElType}(undef, length(bc)), mask))
end

Broadcast.broadcast_unalias(dest::MaskedArray, src::MaskedArray) = Base.unalias(dest, src)
function _copyto!(f, dest::MaskedArray, args...)
    if identical_mask(dest, args...)
        broadcast!(f, getdata(dest), map(getdata, args)...)
    else
        bc = broadcasted(f, args...)
        bc′ = preprocess(dest, bc)
        broadcast!(|, dest.mask, getmask.(args)...) # don't use bc′
        reinit!(dest)
        @inbounds @simd for i in eachindex(bc′)
            if dest.mask[i]
                dest[i] = bc′[i]
            end
        end
    end
end
function Base.copyto!(dest::MaskedArray, bc::Broadcasted{ArrayStyle{MaskedArray}})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bcf = Broadcast.flatten(bc)
    _copyto!(bcf.f, dest, bcf.args...)
    dest
end


struct CDot end
Base.show(io::IO, x::CDot) = print(io, "⋅")

struct ShowMaskedArray{T, N, A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    parent::A
end
Base.size(x::ShowMaskedArray) = size(x.parent)
Base.axes(x::ShowMaskedArray) = axes(x.parent)
@inline function Base.getindex(x::ShowMaskedArray, i::Int...)
    @_propagate_inbounds_meta
    p = x.parent
    p.mask[i...] ? maybecustomshow(p[i...]) : CDot()
end
maybecustomshow(x) = x
maybecustomshow(x::MaskedArray) = ShowMaskedArray(x)

Base.summary(io::IO, x::ShowMaskedArray) = summary(io, x.parent)
Base.show(io::IO, mime::MIME"text/plain", x::MaskedArray) = show(io, mime, ShowMaskedArray(x))
Base.show(io::IO, x::MaskedArray) = show(io, ShowMaskedArray(x))
