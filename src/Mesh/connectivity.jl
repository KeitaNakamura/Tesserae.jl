# Assign dense numbers in storage order. Integer flags can reuse the numbering
# array, and field meshes can supply separate Boolean flags.
function compact_node_numbering!(numbering::AbstractArray{Int}, active=numbering)
    count = 0
    @inbounds for i in eachindex(numbering, active)
        if !iszero(active[i])
            numbering[i] = (count += 1)
        end
    end
    # BitArray can enumerate its set bits without a temporary predicate array.
    indices = active isa AbstractArray{Bool} ? findall(active) : findall(!iszero, numbering)
    indices, numbering
end

function collect_supportnodes(supports)
    nodes = Int[]
    sizehint!(nodes, sum(length, supports; init=0))
    for indices in supports
        append!(nodes, indices)
    end
    unique!(sort!(nodes))
end
