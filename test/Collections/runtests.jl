module CollectionsTest

using Jams.Collections
using Test, Tensorial

@testset "Jams.Collections" begin
    include("AbstractCollection.jl")
    include("LazyCollection.jl")
end

end
