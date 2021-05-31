module CollectionsTest

using Poingr.Collections
using Test

@testset "Poingr.Collections" begin
    include("AbstractCollection.jl")
    include("Collection.jl")
    include("LazyCollection.jl")
end

end
