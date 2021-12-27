"""
    Contact(:sticky)
    Contact(:slip; sep)
    Contact(:friction, coef; sep)

Contact condition handling contact mechanics in MPM.
Following conditions are available:

* `:sticky`: Continuum body is sticked on the boundary surface.
* `:slip`: Continuum body is slipped on the boundary surface.
* `:friction`: Continuum body is slipped with frictional coefficient `coef` on the boundary surface.

If `sep` is `true`, continuum body can leave from the boundary surface.

# Examples
```jldoctest
julia> Contact(:sticky)
Contact(:sticky)

julia> Contact(:slip, sep = true)
Contact(:slip; sep = true)

julia> Contact(:friction, 0.3, sep = true)
Contact(:friction, 0.3; sep = true)
```

---

    (::Contact)(v::Vec, n::Vec)

Compute velocity `v` caused by contact.
The other quantities, which are equivalent to velocity such as momentum and force, are also available.
`n` is the normal unit vector.

# Examples
```jldoctest
julia> contact = Contact(:slip, sep = false);

julia> v = Vec(1.0, -1.0); n = Vec(0.0, -1.0);

julia> v + contact(v, n)
2-element Vec{2, Float64}:
 1.0
 0.0
```
"""
struct Contact
    cond::Symbol
    sep::Bool
    coef::Float64
end

Contact_sticky(cond) = Contact(cond, false, Inf)
Contact_slip(cond; sep = false) = Contact(cond, sep, 0.0)
Contact_friction(cond, coef; sep = false) = Contact(cond, sep, coef)
function Contact(cond::Symbol, args...; kwargs...)
    cond == :sticky   && return Contact_sticky(cond, args...; kwargs...)
    cond == :slip     && return Contact_slip(cond, args...; kwargs...)
    cond == :friction && return Contact_friction(cond, args...; kwargs...)
    throw(ArgumentError("Contact condition `$(QuoteNode(cond))` is not supported"))
end

iscondition(contact::Contact, cond::Symbol) = contact.cond === cond
separation(contact::Contact) = (@assert !iscondition(contact, :sticky); contact.sep)
getfriction(contact::Contact) = (@assert iscondition(contact, :friction); contact.coef)

function Base.show(io::IO, contact::Contact)
    contact.cond == :sticky   && return print(io, "Contact(:sticky)")
    contact.cond == :slip     && return print(io, "Contact(:slip; sep = $(contact.sep))")
    contact.cond == :friction && return print(io, "Contact(:friction, $(contact.coef); sep = $(contact.sep))")
    throw(ArgumentError("Contact condition `$(QuoteNode(contact.cond))` is not supported"))
end

function (contact::Contact)(v::Vec{dim, T}, n::Vec)::Vec{dim, T} where {dim, T}
    v_sticky = -v # contact force for sticky contact
    iscondition(contact, :sticky) && return v_sticky
    d = (v_sticky ⋅ n)
    vn = d * n
    iscondition(contact, :slip) &&
        return d < 0 || !separation(contact) ? vn : zero(vn)
    vt = v_sticky - vn
    if iscondition(contact, :friction)
        if d < 0
            μ = T(getfriction(contact))
            iszero(μ) && return vn # this is necessary since `norm(vt)` can be zero
            return vn + min(1, μ * norm(vn)/norm(vt)) * vt # put `norm(vt)` inside of `min` to handle with deviding zero
        else
            return !separation(contact) ? vn : zero(vn)
        end
    end
end

(contact::Contact)(v, n) = lazy(contact, v, n)
