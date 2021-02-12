# colon computation
## helpers
which_type(::Vararg{Type{<: AbstractCollection{2}}}) = AbstractCollection{2}
which_type(::Vararg{Type{<: UnionGridState}}) = UnionGridState
which_type(::Vararg) = error()
## colon computation
(::Colon)(op, x::Union{AbstractCollection{2}, UnionGridState}) = lazy(op, x)
(::Colon)(op, xs::Tuple) = lazy(op, xs...)
# @generated function (::Colon)(op, xs::Tuple)
    # t = which_type([x for x in xs.parameters if x <: Union{AbstractCollection{2}, UnionGridState}]...)
    # if t <: AbstractCollection{2}
        # exps = [x <: AbstractCollection{2} ? :(xs[$i]) : :(Ref(xs[$i])) for (i,x) in enumerate(xs.parameters)]
        # return :(lazy(broadcasted(op, $(exps...))))
    # else t <: UnionGridState
        # states = [:(xs[$i]) for (i,x) in enumerate(xs.parameters) if x <: UnionGridState]
        # exps = [x <: UnionGridState ? :(_collection(nonzeros(xs[$i]))) : :(Ref(xs[$i])) for (i,x) in enumerate(xs.parameters)]
        # return :(GridStateOperation(indices($(states...)), dofindices($(states...)), LazyCollection{1}(broadcasted(op, $(exps...)))))
    # end
# end

