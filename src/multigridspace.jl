struct MultigridSpace{G <: Grid}
    grids::Vector{G}
end

function MultigridSpace(grid::Grid)
    grids = map(i->copy(grid), 1:Threads.nthreads())
    MultigridSpace(grids)
end
function MultigridSpace(grid::SpGrid)
    grids = map(1:Threads.nthreads()) do i
        spinds = get_spinds(grid)
        data = map(Base.tail(propertynames(grid))) do name
            T = eltype(getproperty(grid, name))
            SpArray{T}(spinds)
        end
        StructArray{eltype(grid)}((get_mesh(grid), data...))
    end
    MultigridSpace{typeof(grid)}(grids)
end

function reinit!(space::MultigridSpace{G}, grid::G, ::Val{names}) where {G <: Grid, names}
    for name in names
        foreach(space.grids) do grid
            fillzero!(getproperty(grid, name))
        end
    end
end

function reinit!(space::MultigridSpace{G}, grid::G, ::Val{names}) where {G <: SpGrid, names}
    for name in Base.tail(propertynames(grid))
        foreach(space.grids) do g
            resize!(get_data(getproperty(g, name)), length(get_data(getproperty(grid, name))))
        end
    end
    for name in names
        foreach(space.grids) do g
            fillzero!(getproperty(g, name))
        end
    end
end

function add!(grid::G, space::MultigridSpace{G}, ::Val{names}) where {G <: Grid, names}
    for name in names
        broadcast!(+, getproperty(grid, name),
                   getproperty(grid, name),
                   map(g -> getproperty(g, name), space.grids)...)
    end
end
