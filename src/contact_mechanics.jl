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
    coef::Float64
    sep::Bool
    thresh::Float64
end

Contact_sticky() = Contact(Inf, false, 0.0)
Contact_slip(; sep = false) = Contact(0.0, sep, 0.0)
Contact_friction(coef; sep = false, thresh = 0.0) = Contact(coef, sep, thresh)
function Contact(cond::Symbol, args...; kwargs...)
    cond == :sticky   && return Contact_sticky(args...; kwargs...)
    cond == :slip     && return Contact_slip(args...; kwargs...)
    cond == :friction && return Contact_friction(args...; kwargs...)
    throw(ArgumentError("Contact condition `:$cond` is not supported"))
end

issticky(contact::Contact) = contact.coef === Inf # Inf is a special value for sticky
isslip(contact::Contact) = contact.coef === 0.0 && contact.thresh === 0.0
isfriction(contact::Contact) = !issticky(contact) && !isslip(contact)

separation(contact::Contact) = (@assert !issticky(contact); contact.sep)

function Base.show(io::IO, contact::Contact)
    issticky(contact)   && return print(io, "Contact(:sticky)")
    isslip(contact)     && return print(io, "Contact(:slip; sep = $(contact.sep))")
    isfriction(contact) && return print(io, "Contact(:friction, $(contact.coef); sep = $(contact.sep), thresh = $(contact.thresh))")
    error("unreachable")
end

function (contact::Contact)(v::Vec{dim, T}, n::Vec{dim, T})::Vec{dim, T} where {dim, T}
    v_sticky = -v # contact force for sticky contact
    issticky(contact) && return v_sticky
    d = (v_sticky ⋅ n)
    vn = d * n
    isslip(contact) && return ifelse(d < 0 || !separation(contact), vn, zero(vn))
    vt = v_sticky - vn
    if isfriction(contact)
        if d < 0
            μ = T(contact.coef)
            c = T(contact.thresh)
            return vn + min(1, (c + μ*norm(vn))/norm(vt)) * vt # put `norm(vt)` inside of `min` to handle with deviding zero
        else
            return ifelse(!separation(contact), vn, zero(vn))
        end
    end
end

(contact::Contact)(v::Vec, n::Vec) = contact(promote(v, n)...)
