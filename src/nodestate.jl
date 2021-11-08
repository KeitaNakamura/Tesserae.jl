struct DefaultNodeState{dim, T}
    m::T
    f::Vec{dim, T}
    v::Vec{dim, T}
    v_n::Vec{dim, T}
    tr_∇v::T
end

default_nodestate_type(::BSpline, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeState{dim, T}
default_nodestate_type(::GIMP, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeState{dim, T}

struct DefaultNodeStateWLS{dim, T}
    m::T
    w::T
    f::Vec{dim, T}
    v::Vec{dim, T}
    tr_∇v::T
end

default_nodestate_type(::WLS, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeStateWLS{dim, T}
