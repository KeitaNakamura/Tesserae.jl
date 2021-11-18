struct DefaultNodeState{dim, T, L, LL}
    m::T
    f::Vec{dim, T}
    v::Vec{dim, T}
    v_n::Vec{dim, T}
    poly_coef::Vec{L, T}
    poly_mat::Mat{L, L, T, LL}
end

default_nodestate_type(::BSpline, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeState{dim, T, dim+1, (dim+1)*(dim+1)}
default_nodestate_type(::GIMP, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeState{dim, T, dim+1, (dim+1)*(dim+1)}

struct DefaultNodeStateWLS{dim, T, L, LL}
    m::T
    w::T
    f::Vec{dim, T}
    v::Vec{dim, T}
    poly_coef::Vec{L, T}
    poly_mat::Mat{L, L, T, LL}
end

default_nodestate_type(::WLS, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeStateWLS{dim, T, dim+1, (dim+1)*(dim+1)}
