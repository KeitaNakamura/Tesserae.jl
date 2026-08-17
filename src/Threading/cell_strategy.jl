# -----------------------------------------------------------------------------
#  CellStrategy
# -----------------------------------------------------------------------------

struct CellStrategy <: PartitionStrategy
    threadsafe_groups::Vector{Vector{Int}}
    region_scratch::RegionScratch{Vector{Int}}
end

threadsafe_groups(cs::CellStrategy) = cs.threadsafe_groups

function CellStrategy(mesh::AbstractCellMesh)
    g = _cell_conflict_graph(mesh)

    coloring = Graphs.degree_greedy_color(g)

    groups = [Int[] for _ in 1:coloring.num_colors]
    @inbounds for (cellid, cell) in enumerate(cells(mesh))
        push!(groups[coloring.colors[cellid]], cellid)
    end

    CellStrategy(groups, RegionScratch{Vector{Int}}())
end

function _cell_conflict_graph(mesh::AbstractCellMesh)
    nc = ncells(mesh)
    nn = length(mesh)
    graph = SimpleGraph(nc)

    node2cells = [Int[] for _ in 1:nn]
    @inbounds for (cellid, cell) in enumerate(cells(mesh))
        for i in supportnodes(mesh, cell)
            push!(node2cells[i], cellid)
        end
    end

    for cells in node2cells
        m = length(cells)
        @inbounds for i in 1:m-1
            cell = cells[i]
            for j in i+1:m
                add_edge!(graph, cell, cells[j])
            end
        end
    end

    graph
end
