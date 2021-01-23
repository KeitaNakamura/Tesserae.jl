function check_lazy(x::LazyCollection, ElType, ans)
    for i in eachindex(x)
        @test (@inferred x[i])::ElType ≈ ans[i]
    end
end

@testset "LazyCollection" begin
    # rank=1
    struct Rank1Collection{T} <: AbstractCollection{1}
        data::Vector{T}
    end
    Base.getindex(x::Rank1Collection, i::Int) = x.data[i]
    Base.setindex!(x::Rank1Collection, v, i::Int) = x.data[i] = v
    Base.length(x::Rank1Collection) = length(x.data)
    # rank=2
    struct Rank2Collection{T} <: AbstractCollection{1}
        data::Vector{T}
    end
    Base.getindex(x::Rank2Collection, i::Int) = x.data[i]
    Base.setindex!(x::Rank2Collection, v, i::Int) = x.data[i] = v
    Base.length(x::Rank2Collection) = length(x.data)

    @testset "Standard methods" begin
        @testset "binary" begin
            for T in (Float32, Float64)
                data = [rand(Vec{3, T}) for i in 1:3]
                x = Rank1Collection(data)
                v = rand(Vec{3, T})
                check_lazy((@inferred x + x)::LazyCollection{1}, Vec{3, T}, data .+ data)
                check_lazy((@inferred x - x)::LazyCollection{1}, Vec{3, T}, data .- data)
                check_lazy((@inferred 2 * x)::LazyCollection{1}, Vec{3, T}, 2 .* data)
                check_lazy((@inferred x * 2)::LazyCollection{1}, Vec{3, T}, data .* 2)
                check_lazy((@inferred x / 2)::LazyCollection{1}, Vec{3, T}, data ./ 2)
                check_lazy((@inferred x ⋅ v)::LazyCollection{1}, T, data .⋅ Ref(v))
                check_lazy((@inferred v ⋅ x)::LazyCollection{1}, T, data .⋅ Ref(v))
                check_lazy((@inferred x ⊗ v)::LazyCollection{1}, Mat{3,3, T}, data .⊗ Ref(v))
                check_lazy((@inferred v ⊗ x)::LazyCollection{1}, Mat{3,3, T}, Ref(v) .⊗ data)
                # should be error?
                check_lazy((@inferred x + v)::LazyCollection{1}, Vec{3, T}, data .+ Ref(v))
                check_lazy((@inferred v + x)::LazyCollection{1}, Vec{3, T}, Ref(v) .+ data)
                check_lazy((@inferred x - v)::LazyCollection{1}, Vec{3, T}, data .- Ref(v))
                check_lazy((@inferred v - x)::LazyCollection{1}, Vec{3, T}, Ref(v) .- data)
            end
        end
        @testset "unary" begin
            for T in (Float32, Float64)
                data = [rand(Mat{3,3, T}) for i in 1:3]
                x = Rank1Collection(data)
                check_lazy((@inferred symmetric(x))::LazyCollection{1}, Tensor{Tuple{@Symmetry{3,3}}, T}, symmetric.(data))
                check_lazy((@inferred tr(x))::LazyCollection{1}, T, tr.(data))
                check_lazy((@inferred vol(x))::LazyCollection{1}, Mat{3,3, T}, vol.(data))
                check_lazy((@inferred mean(x))::LazyCollection{1}, T, mean.(data))
                check_lazy((@inferred det(x))::LazyCollection{1}, T, det.(data))
                # check_lazy((@inferred Tensor2D(x))::LazyCollection{1}, Mat{2,2, T}, getindex.(data, Ref(1:2), Ref(1:2)))
            end
        end
    end
end
