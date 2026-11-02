# Only arrays need an adapter. Scalar indexing and non-array collections keep
# their normal behavior through Base's view/getindex selection.
common_source(x) = x
common_source(A::AbstractArray) = common(A)

function common_refs(ex)
    ex isa Expr || return ex
    ex.head in (:quote, :inert, :$) && return ex
    if ex.head === :ref
        # A function value, as used by @views itself, keeps this helper scalar
        # under @. while allowing nested array expressions to be transformed.
        array = Expr(:call, common_source, common_refs(ex.args[1]))
        indices = map(ex.args[2:end]) do index
            # Index arithmetic is metadata, evaluated normally rather than
            # broadcast. The rest of the equation is delegated to Base.@.
            Expr(:$, index)
        end
        return Expr(:ref, array, indices...)
    end
    Expr(ex.head, map(common_refs, ex.args)...)
end

"""
    @stencil equation
    @stencil begin
        equations...
    end

Select a common valid range and apply `@views` and `@.` to numerical update
equations. Region indices use expression-wide common-range selection; ordinary
scalar and range indices retain their usual bounds rules. Index expressions
are evaluated normally.

Equations in a block execute in order. Each equation selects its own common
range, so a block can mix cell, face, and boundary updates. Unwritten positions
retain their previous values; boundary conditions are supplied by the caller.
The block groups array updates and follows the assignment rules of `@.`:
`=` becomes `.=` and `+=` becomes `.+=`. Prepare scalar bindings, array
allocations, and standalone helper calls outside the block. For example,
`α = 0.1` defines a scalar before entering the block, not inside it.

```julia
@stencil begin
    flux[F] = -α*(temperature[F+e/2] - temperature[F-e/2])/h
    flux[L] = -α*temperature[L+e/2]/(h/2)
    flux[R] =  α*temperature[R-e/2]/(h/2)
    rate[C] = -(flux[C+e/2] - flux[C-e/2])/h
end
```

As with `@.`, use `\$call(...)` to keep a function call scalar. The macro only
sees indexing written in the equations, not indexing hidden inside functions.
The destination must use a Region index when the inputs use common regions.
"""
macro stencil(ex)
    dotted = Expr(:macrocall, GlobalRef(Base.Broadcast, Symbol("@__dot__")), __source__, common_refs(ex))
    esc(Expr(:macrocall, GlobalRef(Base, Symbol("@views")), __source__, dotted))
end
