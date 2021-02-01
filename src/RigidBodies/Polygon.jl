struct Polygon{dim, T} <: RigidBody{dim, T}
    coordinates::Vector{Vec{dim, T}}
end

function Rectangle(bottomleft::Vec{2}, topright::Vec{2}) where {T}
    x0 = bottomleft[1]
    y0 = bottomleft[2]
    x1 = topright[1]
    y1 = topright[2]
    Polygon([Vec(x0, y0), Vec(x1, y0), Vec(x1, y1), Vec(x0, y1)])
end

# handle end+1 index
@inline function Base.getindex(poly::Polygon, i::Int)
    if i == length(poly) + 1
        @inbounds coordinates(poly)[1]
    else
        @boundscheck checkbounds(poly, i)
        @inbounds coordinates(poly)[i]
    end
end

# https://en.wikipedia.org/wiki/Centroid
function center(poly::Polygon{dim, T}) where {dim, T}
    A = zero(T)
    x_c = zero(T)
    y_c = zero(T)
    for i in 1:length(poly)
        @inbounds begin
            Xᵢ = poly[i]
            Xᵢ₊₁ = poly[i+1]
            xᵢ, yᵢ = Xᵢ[1], Xᵢ[2]
            xᵢ₊₁, yᵢ₊₁ = Xᵢ₊₁[1], Xᵢ₊₁[2]
        end
        a = (xᵢ * yᵢ₊₁ - xᵢ₊₁ * yᵢ)
        x_c += (xᵢ + xᵢ₊₁) * a
        y_c += (yᵢ + yᵢ₊₁) * a
        A += a
    end
    A /= 2
    Vec(x_c/6A, y_c/6A)
end

@inline function getline(poly::Polygon, i::Int)
    @boundscheck checkbounds(poly, i)
    if i == length(poly)
        Line(poly[i], poly[1])
    else
        Line(poly[i], poly[i+1])
    end
end

function Base.eachline(poly::Polygon)
    (@inbounds(getline(poly, i)) for i in eachindex(poly))
end

function isinside(poly::Polygon{2}, x::Vec{2}; include_bounds::Bool = true)
    I = 0
    for line in eachline(poly)
        isonline(line, x) && return include_bounds
        I += ray_casting_to_right(line, x)
    end
    isodd(I)
end

function distance(poly::Polygon{2, T}, x::Vec{2, T}, r::Real) where {T}
    dist = nothing
    norm_dist = T(Inf)
    isincontact = false
    for (i, line) in enumerate(eachline(poly))
        d = distance(line, x, r)
        if d !== nothing
            norm_d = norm(d)
            if norm_d < norm_dist
                dist = d
                norm_dist = norm_d
            end
        end
    end
    dist
end
