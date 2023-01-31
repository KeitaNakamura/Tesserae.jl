# https://discourse.julialang.org/t/multithreaded-broadcast/26786/6
struct ThreadedStyle <: Broadcast.BroadcastStyle end
function dot_threads end
@inline Broadcast.broadcasted(f::typeof(dot_threads), x) = Broadcasted{ThreadedStyle}(identity, (x,))

@inline function _copyto!(dest::AbstractArray, bc::Broadcasted{ThreadedStyle})
    @assert bc.f === identity
    @assert bc.args isa Tuple{Any}
    bc′ = Broadcast.preprocess(dest, only(bc.args))
    @threaded for I in eachindex(bc′)
        dest[I] = bc′[I]
    end
    dest
end

function Base.similar(bc::Broadcasted{ThreadedStyle}, ::Type{ElType}) where {ElType}
    similar(only(bc.args), ElType)
end

@inline function Base.copyto!(dest::AbstractArray, bc::Broadcasted{ThreadedStyle})
    axes(dest) == axes(bc) || Broadcast.throwdm(axes(dest), axes(bc))
    _copyto!(dest, bc)
    dest
end

macro dot_threads(ex)
    for op in (:+, :-, :*, :/)
        if Meta.isexpr(ex, Symbol(op, :(=)))
            ex = Expr(:(=), ex.args[1], Expr(:call, op, ex.args[1], ex.args[2]))
            break
        end
    end
    if Meta.isexpr(ex, :(=))
        ex.args[2] = Expr(:call, :(Marble.dot_threads), ex.args[2])
    else
        ex = Expr(:call, :(Marble.dot_threads), ex)
    end
    esc(Broadcast.__dot__(ex))
end


struct LazyDotArray{T, N, Style, Axes <: Tuple, F, Args <: Tuple} <: AbstractArray{T, N}
    bc::Broadcasted{Style, Axes, F, Args}
end
@inline function LazyDotArray(bc::Broadcasted{Style, Axes, F, Args}) where {N, Style, Axes <: Tuple{Vararg{Any, N}}, F, Args}
    T = Broadcast.combine_eltypes(bc.f, bc.args)
    LazyDotArray{T, N, Style, Axes, F, Args}(bc)
end
@inline LazyDotArray(bc::Broadcasted{<: Any, Nothing}) = LazyDotArray(Broadcast.instantiate(bc))
@inline LazyDotArray(f, args...) = LazyDotArray(broadcasted(f, args...))

Base.axes(A::LazyDotArray) = axes(A.bc)
Base.size(A::LazyDotArray) = map(length, axes(A))
@inline function Base.getindex(A::LazyDotArray{T, N}, i::Vararg{Int, N})::T where {T, N}
    @_propagate_inbounds_meta
    A.bc[i...]
end


struct LazyDotStyle <: Broadcast.BroadcastStyle end
function dot_lazy end
@inline Broadcast.broadcasted(f::typeof(dot_lazy), x) = Broadcasted{LazyDotStyle}(identity, (x,))
@inline Broadcast.materialize(bc::Broadcasted{LazyDotStyle}) = LazyDotArray(only(bc.args))

macro dot_lazy(ex)
    ex = Expr(:call, :(Marble.dot_lazy), ex)
    esc(Broadcast.__dot__(ex))
end
