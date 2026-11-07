# Tensor-product values and derivatives

# For code generators only, where `dim` and `k` are literals: `zero_basis_value`
# in basis-weight storage spells the type out instead, because routing it
# through a call loses inference.
jet_value_type(::Order{1}, dim) = Vec{dim}
jet_value_type(::Order{k}, dim) where {k} = Tensor{Tuple{@Symmetry{fill(dim, k)...}}}

@inline function prod_each_dimension(::Order{0}, vals::Vararg{Tuple, dim}) where {dim}
    tuple_otimes(ntuple(d -> vals[d][1], Val(dim)))
end
@generated function prod_each_dimension(::Order{k}, vals::Vararg{Tuple, dim}) where {k, dim}
    TT = jet_value_type(Order(k), dim)
    v = Array{Expr}(undef, size(TT))
    for I in CartesianIndices(v)
        ex = Expr(:tuple)
        for i in 1:dim
            j = count(==(i), Tuple(I)) + 1
            push!(ex.args, :(vals[$i][$j]))
        end
        v[I] = ex
    end
    quote
        @_inline_meta
        v = $(Expr(:tuple, v[Tensorial.independent_to_component_map(TT)]...))
        map($TT, map(tuple_otimes, v)...)
    end
end
@inline tuple_otimes(x::Tuple) = SArray(⊗(map(Vec, x)...))

@generated function tensor_product_jet_at(::Order{k}, axisjets::NTuple{dim,Any}, offsets::NTuple{dim,Int}) where {k,dim}
    quote
        @_inline_meta
        vals1d = @ntuple $dim d -> (@ntuple $(k+1) a -> axisjets[d][a][offsets[d]])
        @ntuple $(k+1) a -> only(prod_each_dimension(Order(a-1), vals1d...))
    end
end
