@inline krondelta(x::Type{<: Vec{3}}) = ones(x)
@inline krondelta(x::Type{<: Tensor{Tuple{3,3}}}) = one(x)
@inline krondelta(x::Type{<: Tensor{Tuple{@Symmetry{3,3}}}}) = one(x)
@inline krondelta(x::AbstractTensor) = krondelta(typeof(x))
