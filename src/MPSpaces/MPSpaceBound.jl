struct MPSpaceBound
    bounddofs::Vector{Int} # non-flat dofs
end

boundary(space::MPSpace) = MPSpaceBound(space.bounddofs)

struct GridStateBound{S}
    parent::S
    bounddofs::Vector{Int}
end

Base.in(state, space::MPSpaceBound) = GridStateBound(state, space.bounddofs)

function set!(dest::GridState, src::GridStateBound)
    set!(dest, src.parent, src.bounddofs)
    dest
end
