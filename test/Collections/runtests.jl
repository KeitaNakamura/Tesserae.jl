module CollectionsTest

using Jams.Collections
using Test

@testset "Jams.Collections" begin
    include("AbstractCollection.jl")
    include("Collection.jl")
    include("LazyCollection.jl")
end

end
