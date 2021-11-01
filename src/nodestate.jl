struct DefaultNodeState{dim, T}
    m::Float64
    f::Vec{dim, T}
    v::Vec{dim, T}
    v_n::Vec{dim, T}
    dϵ_v::T
end

default_nodestate_type(::BSpline, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeState{dim, T}
default_nodestate_type(::GIMP, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeState{dim, T}

struct DefaultNodeStateWLS{dim, T}
    m::Float64
    w::Float64
    f::Vec{dim, T}
    v::Vec{dim, T}
    dϵ_v::T
end

default_nodestate_type(::WLS, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeStateWLS{dim, T}
