# https://discourse.julialang.org/t/multithreaded-broadcast/26786/6
struct ThreadedStyle <: BroadcastStyle end
function dot_threads end
Broadcast.broadcasted(f::typeof(dot_threads), x) = Broadcasted{ThreadedStyle}(identity, (x,))

@inline function _copyto!(dest::AbstractArray, bc::Broadcasted{ThreadedStyle})
    @assert bc.f === identity
    @assert bc.args isa Tuple{Any}
    bc′ = preprocess(dest, bc.args[1])
    Threads.@threads for I in eachindex(bc′)
        @inbounds dest[I] = bc′[I]
    end
    dest
end

@inline function Base.copyto!(dest::AbstractArray, bc::Broadcasted{ThreadedStyle})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    _copyto!(dest, bc)
end

@inline function _threads_copyto!(f, dest::SpArray, args...)
    if identical_mask(dest, args...)
        bc = broadcasted(f, args...)
        _copyto!(dest, broadcasted(dot_threads, bc))
    else
        bc = broadcasted(f, args...)
        bc′ = preprocess(dest, bc)
        broadcast!(|, dest.mask, getmask.(args)...) # don't use bc′
        reinit!(dest)
        @inbounds Threads.@threads for i in eachindex(bc′)
            if dest.mask[i]
                dest[i] = bc′[i]
            end
        end
    end
end

@inline function Base.copyto!(dest::SpArray, bc::Broadcasted{ThreadedStyle})
    axes(dest) == axes(bc) || throwdm(axes(dest), axes(bc))
    bcf = Broadcast.flatten(bc.args[1])
    _threads_copyto!(bcf.f, dest, bcf.args...)
end

macro dot_threads(ex)
    for op in (:+, :-, :*, :/)
        if Meta.isexpr(ex, Symbol(op, :(=)))
            ex = Expr(:(=), ex.args[1], Expr(:call, op, ex.args[1], ex.args[2]))
            break
        end
    end
    @assert Meta.isexpr(ex, :(=))
    ex.args[2] = Expr(:call, :(Poingr.dot_threads), ex.args[2])
    esc(Broadcast.__dot__(ex))
end


struct LazyDotArray{T, N, F, Args} <: AbstractArray{T, N}
    f::F
    args::Args
end
Base.axes(A::LazyDotArray) = axes(broadcasted(A))
Base.size(A::LazyDotArray) = map(length, axes(A))
@inline Base.broadcasted(A::LazyDotArray) = broadcasted(A.f, A.args...)
@inline Base.getindex(A::LazyDotArray{<: Any, N}, i::Vararg{Int, N}) where {N} = (@_propagate_inbounds_meta; broadcasted(A)[i...])

@inline _LazyDotArray(bc::Broadcasted{<: Any, <: Any, F, Args}, axes::Tuple{Vararg{Any, N}}) where {N, F, Args} =
    LazyDotArray{Broadcast.combine_eltypes(bc.f, bc.args), N, F, Args}(bc.f, bc.args)
@inline LazyDotArray(bc::Broadcasted) = _LazyDotArray(bc, axes(bc))

struct LazyDotStyle <: BroadcastStyle end
function dot_lazy end
@inline Broadcast.broadcasted(f::typeof(dot_lazy), x) = Broadcasted{LazyDotStyle}(identity, (x,))
@inline Broadcast.materialize(bc::Broadcasted{LazyDotStyle}) = LazyDotArray(bc.args[1])

macro dot_lazy(ex)
    ex = Expr(:call, :(Poingr.dot_lazy), ex)
    esc(Broadcast.__dot__(ex))
end
