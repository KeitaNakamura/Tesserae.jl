struct DefaultNodeState{dim, T, L, LL}
    m::T
    v::Vec{dim, T}
    v_n::Vec{dim, T}
    poly_coef::Vec{L, T}
    poly_mat::Mat{L, L, T, LL}
end

default_nodestate_type(::Interpolation, ::Val{dim}, ::Val{T}) where {dim, T} = DefaultNodeState{dim, T, dim+1, (dim+1)*(dim+1)}
