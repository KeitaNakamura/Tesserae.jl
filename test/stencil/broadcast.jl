# A backend with its own view type and broadcast style verifies that execution
# is handed back to the parent's numerical implementation after lowering.
struct TrackedArray{T, N, A<:AbstractArray{T, N}} <: AbstractArray{T, N}
    data::A
    calls::Base.RefValue{Int}
end
Base.size(A::TrackedArray) = size(A.data)
function Base.getindex(::TrackedArray, I::Vararg{Int})
    error("scalar indexing is disabled for this backend")
end
function Base.setindex!(::TrackedArray, v, I::Vararg{Int})
    error("scalar indexing is disabled for this backend")
end
function Base.fill!(A::TrackedArray, value)
    fill!(A.data, value)
    A
end
Base.view(A::TrackedArray, I...) = TrackedArray(view(A.data, I...), A.calls)
function Base.similar(A::TrackedArray, ::Type{T}, dims::Dims) where {T}
    TrackedArray(similar(A.data, T, dims), Ref(0))
end
struct TrackedStyle{N} <: Base.Broadcast.AbstractArrayStyle{N} end
TrackedStyle(::Val{N}) where {N} = TrackedStyle{N}()
TrackedStyle{N}(::Val{M}) where {N, M} = TrackedStyle{M}()
Base.Broadcast.BroadcastStyle(::Type{<:TrackedArray{T, N}}) where {T, N} = TrackedStyle{N}()
untrack(A::TrackedArray) = A.data
untrack(A::OffsetArray) = OffsetArray(untrack(parent(A)), axes(A))
untrack(x::Tuple) = map(untrack, x)
untrack(x::Base.RefValue) = Ref(untrack(x[]))
untrack(x) = x
untrack(bc::Base.Broadcast.Broadcasted) = Base.Broadcast.broadcasted(bc.f, map(untrack, bc.args)...)
function Base.copyto!(A::TrackedArray, bc::Base.Broadcast.Broadcasted{Nothing})
    A.calls[] += 1
    Base.Broadcast.materialize!(A.data, untrack(bc))
    A
end

using Tesserae.Stencil: common

neighbor_sum(A, I) = sum(@view A[I-oneunit(I):I+oneunit(I)])

@testset "Explicit common adapters with standard views and broadcast" begin
    C = Region(Cell(), (5,))
    e = first(unitoffsets(C))
    values = [10., 20., 40., 80., 160.]
    result = fill(-999., 5)
    source, target = common(values), common(result)
    @views @. target[C] = source[C+e] - source[C-e]
    @test result == [-999., 30., 60., 120., -999.]
    @test common(source) === source
    @test parent(view(source, 2:4)) === values
    @test source[1] == values[1]
    @test_throws BoundsError view(values, C+e)
end

@testset "Periodic opposite boundaries and alias handling" begin
    C = Region(Cell(), (5, 4))
    F = Region(Face(1), C)
    ex, ey = unitoffsets(C)
    L, U = boundaries(F)
    RL, RU = L, U
    p = reshape(Float64.(1:20), 5, 4)
    out = fill(-999., size(F))
    @stencil out[RL] = p[RL+ex/2]-p[RU-ex/2]
    @test out[1:1, :] == p[1:1, :]-p[end:end, :]
    @test all(==(-999.), out[2:end, :])

    C1 = Region(Cell(), (5,))
    e = unitoffsets(C1)[1]
    R = C1
    q = [10., 20., 40., 80., 160.]
    @stencil q[R] = q[R+e]-q[R-e]
    @test q == [10., 30., 60., 120., 160.]
end

@testset "Ordinary arguments, scalar assignment, and shape errors" begin
    C = Region(Cell(), (5, 4))
    ex, ey = unitoffsets(C)
    R = C
    p = reshape(Float64.(1:20), 5, 4)
    out = fill(-999., size(p))
    coefficient = reshape(Float64.(1:5), 5, 1)
    @stencil out[R] = sin(p[R+ex])-p[R-ex]*coefficient
    @test out[2:4, :] == sin.(p[3:5, :])-p[1:3, :].*coefficient[2:4, :]
    @test all(==(-999.), out[[1, end], :])
    @stencil out[R] = 2
    @test all(==(2), out)
    @stencil out[R] = coefficient
    @test out == repeat(coefficient, 1, 4)
    out .= 2
    @stencil out[R] += p[R+ex]
    @test out[1:4, :] == 2 .+ p[2:5, :]
    @test all(==(2), out[end, :])
    wrong_shape = zeros(3, 4)
    @test_throws DimensionMismatch (@stencil out[R] = wrong_shape+p[R])
    F = Region(Face(1), C)
    @test_throws DimensionMismatch (@stencil out[F] = p[F])
end

@testset "Empty common regions and ordinary tuple broadcasting" begin
    C = Region(Cell(), (5,))
    e = unitoffsets(C)[1]
    R = C
    p = [10., 20., 40., 80., 160.]
    out = fill(-999., 5)
    @stencil out[R] = p[R+20e]
    @test all(==(-999.), out)
    @stencil out[R] = p[R+e]-p[R-e]+(1., 2., 3., 4., 5.)
    @test out == [-999., 32., 63., 124., -999.]
end

@testset "Delegate to a backend's own broadcast style" begin
    C = Region(Cell(), (5, 4))
    ex, ey = unitoffsets(C)
    R = C
    qdata = reshape(Float64.(1:20), 5, 4)
    q = TrackedArray(qdata, Ref(0))
    out = TrackedArray(fill(-999., 5, 4), Ref(0))
    coefficient = TrackedArray(reshape(Float64.(1:5), 5, 1), Ref(0))
    source, target = common(q), common(out)
    @views @. target[R] = source[R+ex]-source[R-ex]*coefficient
    @test out.data[2:4, :] == qdata[3:5, :]-qdata[1:3, :].*coefficient.data[2:4, :]
    @test all(==(-999.), out.data[[1, end], :])
    @test out.calls[] == 1

    fill!(out.data, -999)
    @stencil out[C] = q[C+ex]-q[C-ex]*coefficient
    @test out.data[2:4, :] == qdata[3:5, :]-qdata[1:3, :].*coefficient.data[2:4, :]
    @test all(==(-999.), out.data[[1, end], :])
    @test out.calls[] == 2
end

module StencilMacroWrappers
macro wrap(ex)
    quote
        let value = $(esc(ex))
            value
        end
    end
end
macro counted(counter, ex)
    quote
        $(esc(counter))[] += 1
        $(esc(ex))
    end
end
end

@testset "A stencil macro wrapping common, views, and dot" begin
    C = Region(Cell(), (5,))
    F = Region(Face(1), C)
    e = unitoffsets(C)[1]
    temperature = [10., 20., 40., 80., 160.]
    flux = fill(NaN, size(F))
    h = [1.0]
    α = 0.1

    @stencil flux[F] =
        -α * (temperature[F+e/2] - temperature[F-e/2]) / h[1]
    @test isequal(flux, [NaN, -1., -2., -4., -8., NaN])
    @test_throws BoundsError view(temperature, C+e)

    # Per-equation range selection within a block, with ordinary boundary views.
    L, U = boundaries(F, 1)
    rate = fill(NaN, size(C))
    fill!(flux, NaN)
    @stencil begin
        flux[F] = -α * (temperature[F+e/2] - temperature[F-e/2]) / h[1]
        flux[L] = -α * temperature[L+e/2] / (h[1]/2)
        flux[U] =  α * temperature[U-e/2] / (h[1]/2)
        rate[C] = -(flux[C+e/2] - flux[C-e/2]) / h[1]
    end
    @test flux == [-2., -1., -2., -4., -8., 32.]
    @test rate == [-1., 1., 2., 4., -40.]

    # Normal indexing, including scalar coefficients and end, is unchanged.
    normal = zeros(5)
    @stencil normal[2:end-1] = temperature[3:end] - temperature[1:end-2]
    @test normal == [0., 30., 60., 120., 0.]
    @test_throws BoundsError (@stencil normal[1:6] = 0)
    @stencil normal[begin+1:end-1] = temperature[begin+2:end] - temperature[begin:end-2]
    @test normal == [0., 30., 60., 120., 0.]

    # Dot's ordinary escape syntax still applies to scalar function calls.
    once = Ref(0)
    coefficient() = (once[] += 1; 2.0)
    @stencil normal[C] = temperature[C] * $coefficient()
    @test normal == 2temperature
    @test once[] == 1

    # Compound update, alias handling and nested array indexing.
    q = copy(temperature)
    @stencil q[C] += q[C+e] - q[C-e]
    @test q == [10., 50., 100., 200., 160.]
    fields = [zeros(size(F)), zeros(size(F))]
    d = 2
    @stencil fields[d][F] = temperature[F+e/2] - temperature[F-e/2]
    @test fields[2] == [0., 10., 20., 40., 80., 0.]
    @test all(iszero, fields[1])
end

@testset "Macros and blocks preserve expression-wide range selection" begin
    C = Region(Cell(), (5,))
    e = only(unitoffsets(C))
    values = [10.0, 20.0, 40.0, 80.0, 160.0]
    result = fill(-999.0, 5)
    expected = [-999.0, 70.0, 140.0, 280.0, -999.0]

    @stencil result[C] = @inbounds 2values[C+e]-values[C-e]
    @test result == expected
    fill!(result, -999)
    @stencil result[C] = Base.@inbounds 2values[C+e]-values[C-e]
    @test result == expected
    fill!(result, -999)
    @stencil result[C] = @fastmath 2values[C+e]-values[C-e]
    @test result == expected

    counter = Ref(0)
    value = -1
    fill!(result, -999)
    @stencil result[C] = StencilMacroWrappers.@counted counter StencilMacroWrappers.@wrap 2values[C+e]-values[C-e]
    @test result == expected
    @test counter[] == 1
    @test value == -1

    fill!(result, -999)
    @stencil result[C] = let upper = 2values[C+e], lower = values[C-e]
        upper-lower
    end
    @test result == expected
    @stencil @inbounds begin
        result[C] = values[C+e]-values[C-e]
        result[C] *= 2
    end
    @test result == [-1998.0, 60.0, 120.0, 240.0, -1998.0]

    alias = copy(values)
    @stencil alias[C] = @inbounds alias[C+e]-alias[C-e]
    @test alias == [10.0, 30.0, 60.0, 120.0, 160.0]

    indices = CartesianIndices(size(C))
    padded = fill(7.0, C; halo=1)
    @stencil result[C] = @inbounds neighbor_sum((padded,), indices[C])
    @test result == fill(21.0, 5)
    @test_throws BoundsError (@stencil result[C] = neighbor_sum((values,), indices[C]))

    source = TrackedArray(copy(values), Ref(0))
    target = TrackedArray(fill(-999.0, 5), Ref(0))
    coefficient = TrackedArray(fill(2.0, 5), Ref(0))
    @stencil target[C] = @inbounds 2source[C+e]-source[C-e]*coefficient
    @test target.data == [-999.0, 60.0, 120.0, 240.0, -999.0]
    @test target.calls[] == 1

    before = copy(result)
    @test_throws ArgumentError (@stencil result = @inbounds values[C+e]-values[C-e])
    @test result == before
    wrong_shape = zeros(3)
    @test_throws DimensionMismatch (@stencil result[C] = @inbounds values[C]+wrong_shape)
    @test result == before
end

@testset "Block ranges, ordering, and unsupported destinations" begin
    C = Region(Cell(), (5,))
    e = unitoffsets(C)[1]
    p = [10., 20., 40., 80., 160.]
    forward, backward = fill(NaN, 5), fill(NaN, 5)
    result = zeros(5)
    @stencil begin
        forward[C] = p[C+e] - p[C]
        backward[C] = p[C] - p[C-e]
        result[C] = forward[C] + backward[C]
    end
    @test isequal(forward, [10., 20., 40., 80., NaN])
    @test isequal(backward, [NaN, 10., 20., 40., 80.])
    @test isequal(result, [NaN, 30., 60., 120., NaN])

    # An ordinary destination must not start writing deferred region inputs.
    fill!(result, -999)
    @test_throws ArgumentError (@stencil result = p[C+e])
    @test all(==(-999), result)

    # Helpers are resolved in the library, not in the caller's local scope.
    let common_source = nothing, common = nothing
        @stencil result[C] = p[C+e] - p[C-e]
    end
    @test result == [-999., 30., 60., 120., -999.]
end

@testset "Empty overlap on a singleton axis" begin
    for n in ((1,), (1, 3), (3, 1))
        C = Region(Cell(), n)
        e = unitoffsets(C)[findfirst(==(1), n)]
        a = fill(7, C)
        @stencil a[C+e] = 0
        @stencil a[C] = error(a[C-e])
        @test all(==(7), a)
    end
end
