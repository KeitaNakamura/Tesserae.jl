"""
    Line(a::Vec, b::Vec)
    Line(a::Vec => b::Vec)
"""
struct Line{dim, T} <: RigidBody{dim, T}
    coordinates::Vector{Vec{dim, T}}
    function Line{dim, T}(coordinates::Vector{Vec{dim, T}}) where {dim, T}
        @assert length(coordinates) == 2
        new{dim, T}(coordinates)
    end
end
Line(coordinates::Vector{Vec{dim, T}}) where {dim, T} = Line{dim, T}(coordinates)

Line(a::Vec, b::Vec) = Line([a, b])
Line(pair::Pair) = Line(pair.first, pair.second)

center(line::Line) = sum(line) / 2

function _distance(line::Line, x::Vec)
    v = line[2] - line[1]
    a_to_x = x - line[1]
    scale = (a_to_x ⋅ v) / (v ⋅ v)
    distance = scale*v - a_to_x
    distance, scale
end

"""
    distance(::Line, x::Vec)
    distance(::Line, x::Vec, threshold::Real)

Compute the distance vector from `x` to perpendicular foot.
When `threshold` is given, check the contact between line and point `x`,
and return `nothing` if contact is not detected.
Note that if the perpendicular foot does not lie on the line,
contact detection is performed using distance between `x` and vertices of line.

```jldoctest
julia> line = Line(@Vec[0.0, 0.0] => @Vec[1.0, 1.0])
2-element Line{2,Float64}:
 [0.0, 0.0]
 [1.0, 1.0]

julia> distance(line, @Vec[1.0, 0.0])
2-element Tensor{Tuple{2},Float64,1,2}:
 -0.5
  0.5

julia> distance(line, @Vec[1.0, 0.0], 0.5) === nothing
true

julia> distance(line, @Vec[1.0, 1.5], 1.0) === nothing
true
```
"""
function distance(line::Line, x::Vec)
    _distance(line, x)[1]
end
function distance(line::Line, x::Vec, r::Real)
    r² = r^2
    d, scale = _distance(line, x)
    d ⋅ normalunit(line) > 0 && return nothing
    if 0 ≤ scale ≤ 1 # perpendicular foot is on line
        (d ⋅ d) ≤ r² && return d
    else
        @inbounds begin
            x_to_a = line[1] - x
            x_to_b = line[2] - x
        end
        (x_to_a ⋅ x_to_a) ≤ r² && return x_to_a
        (x_to_b ⋅ x_to_b) ≤ r² && return x_to_b
    end
    nothing
end

"""
    perpendicularfoot(::Line, x::Vec)

Compute the position of perpendicular foot.
"""
function perpendicularfoot(line::Line, x::Vec)
    x + distance(line, x)
end

function normalunit(line::Line{2})
    @inbounds begin
        v = line[2] - line[1]
        n = Vec(v[2], -v[1])
    end
    n / norm(n)
end

"""
    RigidBodies.isonline(line::Line, x::Vec)

Return `true` if `x` is on `line`.

# Examples
```jldoctest
julia> line = Line(@Vec[0.0, 0.0], @Vec[2.0, 2.0])
2-element Line{2,Float64}:
 [0.0, 0.0]
 [2.0, 2.0]

julia> RigidBodies.isonline(line, @Vec[1.0, 1.0])
true

julia> RigidBodies.isonline(line, @Vec[1.0, 0.0])
false
```
"""
function isonline(line::Line, x::Vec)
    d, scale = _distance(line, x)
    0 ≤ scale ≤ 1 && (d ⋅ d) < eps(eltype(d))
end
# much faster for 2D
function isonline(line::Line{2}, X::Vec{2})
    @inbounds begin
        x, y = X[1], X[2]
        a, b = line
        a_x, a_y = a[1], a[2]
        b_x, b_y = b[1], b[2]
    end
    if (a_x ≤ x ≤ b_x) || (b_x ≤ x ≤ a_x)
        if (a_y ≤ y ≤ b_y) || (b_y ≤ y ≤ a_y)
            (x - a_x) * (b_y - a_y) == (y - a_y) * (b_x - a_x) && return true
        end
    end
    false
end

# helper function for `isinside`
function ray_casting_to_right(line::Line{2}, X::Vec{2})
    @inbounds begin
        x, y = X[1], X[2]
        a, b = line
        a_x, a_y = a[1], a[2]
        b_x, b_y = b[1], b[2]
    end
    if (a_y ≤ y < b_y) || # upward case
       (b_y ≤ y < a_y)    # downward case
        x_line = a_x + (y - a_y) / (b_y - a_y) * (b_x - a_x)
        x < x_line && return true
    end
    false
end
