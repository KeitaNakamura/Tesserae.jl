# -----------------------------------------------------------------------------
#  Cell shapes
# -----------------------------------------------------------------------------

# Cell shapes for FEM and IGA meshes.
abstract type Shape{dim} end

localnodes(s::Shape) = localnodes(Float64, s)

nlocalnodes(s::Shape) = length(localnodes(s))
nsupportnodes(s::Shape) = nlocalnodes(s)

get_dimension(::Shape{dim}) where {dim} = dim

# ---- Line ----

abstract type Line <: Shape{1} end

"""
    Line2()

# Geometry
```
      η
      ^
      |
      |
1-----+-----2 --> ξ
```
"""
struct Line2 <: Line end

get_order(::Line2) = Order(1)
primarynodes_indices(::Line2) = SOneTo(2)

function localnodes(::Type{T}, ::Line2) where {T}
    SVector{2, Vec{1, T}}(
        (-1.0,),
        ( 1.0,),
    )
end

function value(::Line2, X::Vec{1, T}) where {T}
    ξ = X[1]
    SVector{2, T}(
        0.5 * (1-ξ),
        0.5 * (1+ξ),
    )
end

"""
    Line3()

# Geometry
```
      η
      ^
      |
      |
1-----3-----2 --> ξ
```
"""
struct Line3 <: Line end

get_order(::Line3) = Order(2)
primarynodes_indices(::Line3) = SOneTo(2)

function localnodes(::Type{T}, ::Line3) where {T}
    SVector{3, Vec{1, T}}(
        (-1.0,),
        ( 1.0,),
        ( 0.0,),
    )
end

function value(::Line3, X::Vec{1, T}) where {T}
    ξ = X[1]
    SVector{3, T}(
        -0.5 * ξ*(1-ξ),
         0.5 * ξ*(1+ξ),
        1 - ξ^2,
    )
end

"""
    Line4()

# Geometry
```
      η
      ^
      |
      |
1---3---4---2 --> ξ
```
"""
struct Line4 <: Line end

get_order(::Line4) = Order(3)
primarynodes_indices(::Line4) = SOneTo(2)

function localnodes(::Type{T}, ::Line4) where {T}
    SVector{4, Vec{1, T}}(
        (-1.0,),
        ( 1.0,),
        (-1/3,),
        ( 1/3,),
    )
end

function value(::Line4, X::Vec{1, T}) where {T}
    ξ = X[1]
    SVector{4, T}(
        -0.5625 * (ξ+1/3)*(ξ-1/3)*(ξ-1),
         0.5625 * (ξ+1)*(ξ+1/3)*(ξ-1/3),
         1.6875 * (ξ+1)*(ξ-1/3)*(ξ-1),
        -1.6875 * (ξ+1)*(ξ+1/3)*(ξ-1),
    )
end

# ---- Quad ----

abstract type Quad <: Shape{2} end

"""
    Quad4()

# Geometry
```
      η
      ^
      |
4-----------3
|     |     |
|     |     |
|     +---- | --> ξ
|           |
|           |
1-----------2
```
"""
struct Quad4 <: Quad end

get_order(::Quad4) = Order(1)
primarynodes_indices(::Quad4) = SOneTo(4)

faceshape(::Quad4) = Line2()
faces(::Quad4) = SVector(SVector(1,2), SVector(2,3), SVector(3,4), SVector(4,1))

function localnodes(::Type{T}, ::Quad4) where {T}
    SVector{4, Vec{2, T}}(
        (-1.0, -1.0),
        ( 1.0, -1.0),
        ( 1.0,  1.0),
        (-1.0,  1.0),
    )
end

function value(::Quad4, X::Vec{2, T}) where {T}
    ξ, η = X
    SVector{4, T}(
        0.25 * (1-ξ) * (1-η),
        0.25 * (1+ξ) * (1-η),
        0.25 * (1+ξ) * (1+η),
        0.25 * (1-ξ) * (1+η),
    )
end

"""
    Quad8()

# Geometry
```
      η
      ^
      |
4-----7-----3
|     |     |
|     |     |
8     +---- 6 --> ξ
|           |
|           |
1-----5-----2
```
"""
struct Quad8 <: Quad end

get_order(::Quad8) = Order(2)
primarynodes_indices(::Quad8) = SOneTo(4)

faceshape(::Quad8) = Line3()
faces(::Quad8) = SVector(SVector(1,2,5), SVector(2,3,6), SVector(3,4,7), SVector(4,1,8))

function localnodes(::Type{T}, ::Quad8) where {T}
    SVector{8, Vec{2, T}}(
        (-1.0, -1.0),
        ( 1.0, -1.0),
        ( 1.0,  1.0),
        (-1.0,  1.0),
        ( 0.0, -1.0),
        ( 1.0,  0.0),
        ( 0.0,  1.0),
        (-1.0,  0.0),
    )
end

function value(::Quad8, X::Vec{2, T}) where {T}
    ξ, η = X
    SVector{8, T}(
        0.25 * (1-ξ) * (1-η) * (-ξ-η-1),
        0.25 * (1+ξ) * (1-η) * ( ξ-η-1),
        0.25 * (1+ξ) * (1+η) * ( ξ+η-1),
        0.25 * (1-ξ) * (1+η) * (-ξ+η-1),
        0.5  * (1-ξ^2) * (1-η),
        0.5  * (1+ξ) * (1-η^2),
        0.5  * (1-ξ^2) * (1+η),
        0.5  * (1-ξ) * (1-η^2),
    )
end

"""
    Quad9()

# Geometry
```
      η
      ^
      |
4-----7-----3
|     |     |
|     |     |
8     9---- 6 --> ξ
|           |
|           |
1-----5-----2
```
"""
struct Quad9 <: Quad end

get_order(::Quad9) = Order(2)
primarynodes_indices(::Quad9) = SOneTo(4)

faceshape(::Quad9) = Line3()
faces(::Quad9) = SVector(SVector(1,2,5), SVector(2,3,6), SVector(3,4,7), SVector(4,1,8))

function localnodes(::Type{T}, ::Quad9) where {T}
    SVector{9, Vec{2, T}}(
        (-1.0, -1.0),
        ( 1.0, -1.0),
        ( 1.0,  1.0),
        (-1.0,  1.0),
        ( 0.0, -1.0),
        ( 1.0,  0.0),
        ( 0.0,  1.0),
        (-1.0,  0.0),
        ( 0.0,  0.0),
    )
end

function value(::Quad9, X::Vec{2, T}) where {T}
    ξ, η = X
    SVector{9, T}(
         0.25 * ξ * η * (1-ξ) * (1-η),
        -0.25 * ξ * η * (1+ξ) * (1-η),
         0.25 * ξ * η * (1+ξ) * (1+η),
        -0.25 * ξ * η * (1-ξ) * (1+η),
        -0.5 * η * (1-ξ^2) * (1-η),
         0.5 * ξ * (1+ξ) * (1-η^2),
         0.5 * η * (1-ξ^2) * (1+η),
        -0.5 * ξ * (1-ξ) * (1-η^2),
         (1-ξ^2) * (1-η^2),
    )
end

# ---- Hex ----

abstract type Hex <: Shape{3} end

@doc raw"""
    Hex8()

# Geometry
```
       η
4----------3
|\     ^   |\
| \    |   | \
|  \   |   |  \
|   8------+---7
|   |  +-- |-- | -> ξ
1---+---\--2   |
 \  |    \  \  |
  \ |     \  \ |
   \|      ζ  \|
    5----------6
```
"""
struct Hex8 <: Hex end

get_order(::Hex8) = Order(1)
primarynodes_indices(::Hex8) = SOneTo(8)

faceshape(::Hex8) = Quad4()
faces(::Hex8) = SVector(SVector(1,5,8,4), SVector(6,2,3,7), SVector(1,2,6,5), SVector(3,4,8,7), SVector(2,1,4,3), SVector(5,6,7,8))

function localnodes(::Type{T}, ::Hex8) where {T}
    SVector{8, Vec{3, T}}(
        (-1.0, -1.0, -1.0),
        ( 1.0, -1.0, -1.0),
        ( 1.0,  1.0, -1.0),
        (-1.0,  1.0, -1.0),
        (-1.0, -1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0,  1.0),
        (-1.0,  1.0,  1.0),
    )
end

function value(::Hex8, X::Vec{3, T}) where {T}
    ξ, η, ζ = X
    SVector{8, T}(
        0.125 * (1-ξ) * (1-η) * (1-ζ),
        0.125 * (1+ξ) * (1-η) * (1-ζ),
        0.125 * (1+ξ) * (1+η) * (1-ζ),
        0.125 * (1-ξ) * (1+η) * (1-ζ),
        0.125 * (1-ξ) * (1-η) * (1+ζ),
        0.125 * (1+ξ) * (1-η) * (1+ζ),
        0.125 * (1+ξ) * (1+η) * (1+ζ),
        0.125 * (1-ξ) * (1+η) * (1+ζ),
    )
end

@doc raw"""
    Hex20()

# Geometry
```
       η
       ^
4----14+---3
|\     |   |\
|16    |   | 15
10 \   |   12 \
|   8----20+---7
|   |  +-- |-- | -> ξ
1---+-9-\--2   |
 \ 18    \  \  19
 11 |     \  13|
   \|      ζ  \|
    5----17----6
```
"""
struct Hex20 <: Hex end

get_order(::Hex20) = Order(2)
primarynodes_indices(::Hex20) = SOneTo(8)

faceshape(::Hex20) = Quad8()
faces(::Hex20) = SVector(SVector(1,5,8,4,11,18,16,10), SVector(6,2,3,7,13,12,15,19), SVector(1,2,6,5,9,13,17,11), SVector(3,4,8,7,14,16,20,15), SVector(2,1,4,3,9,10,14,12), SVector(5,6,7,8,17,19,20,18))

function localnodes(::Type{T}, ::Hex20) where {T}
    SVector{20, Vec{3, T}}(
        (-1.0, -1.0, -1.0),
        ( 1.0, -1.0, -1.0),
        ( 1.0,  1.0, -1.0),
        (-1.0,  1.0, -1.0),
        (-1.0, -1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0,  1.0),
        (-1.0,  1.0,  1.0),
        ( 0.0, -1.0, -1.0),
        (-1.0,  0.0, -1.0),
        (-1.0, -1.0,  0.0),
        ( 1.0,  0.0, -1.0),
        ( 1.0, -1.0,  0.0),
        ( 0.0,  1.0, -1.0),
        ( 1.0,  1.0,  0.0),
        (-1.0,  1.0,  0.0),
        ( 0.0, -1.0,  1.0),
        (-1.0,  0.0,  1.0),
        ( 1.0,  0.0,  1.0),
        ( 0.0,  1.0,  1.0),
    )
end

function value(::Hex20, X::Vec{3, T}) where {T}
    ξ, η, ζ = X
    SVector{20, T}(
        -0.125 * (1-ξ) * (1-η) * (1-ζ) * (2+ξ+η+ζ),
        -0.125 * (1+ξ) * (1-η) * (1-ζ) * (2-ξ+η+ζ),
        -0.125 * (1+ξ) * (1+η) * (1-ζ) * (2-ξ-η+ζ),
        -0.125 * (1-ξ) * (1+η) * (1-ζ) * (2+ξ-η+ζ),
        -0.125 * (1-ξ) * (1-η) * (1+ζ) * (2+ξ+η-ζ),
        -0.125 * (1+ξ) * (1-η) * (1+ζ) * (2-ξ+η-ζ),
        -0.125 * (1+ξ) * (1+η) * (1+ζ) * (2-ξ-η-ζ),
        -0.125 * (1-ξ) * (1+η) * (1+ζ) * (2+ξ-η-ζ),
        0.25 * (1-ξ) * (1+ξ) * (1-η) * (1-ζ),
        0.25 * (1-η) * (1+η) * (1-ξ) * (1-ζ),
        0.25 * (1-ζ) * (1+ζ) * (1-ξ) * (1-η),
        0.25 * (1-η) * (1+η) * (1+ξ) * (1-ζ),
        0.25 * (1-ζ) * (1+ζ) * (1+ξ) * (1-η),
        0.25 * (1-ξ) * (1+ξ) * (1+η) * (1-ζ),
        0.25 * (1-ζ) * (1+ζ) * (1+ξ) * (1+η),
        0.25 * (1-ζ) * (1+ζ) * (1-ξ) * (1+η),
        0.25 * (1-ξ) * (1+ξ) * (1-η) * (1+ζ),
        0.25 * (1-η) * (1+η) * (1-ξ) * (1+ζ),
        0.25 * (1-η) * (1+η) * (1+ξ) * (1+ζ),
        0.25 * (1-ξ) * (1+ξ) * (1+η) * (1+ζ),
    )
end

@doc raw"""
    Hex27()

# Geometry
```
       η
       ^
4----14+---3
|\     |   |\
|16    25  | 15
10 \ 21|   12 \
|   8----20+---7
|23 |  27--|-24| -> ξ
1---+-9-\--2   |
 \ 18    26 \  19
 11 |  22 \  13|
   \|      ζ  \|
    5----17----6
```
"""
struct Hex27 <: Hex end

get_order(::Hex27) = Order(2)
primarynodes_indices(::Hex27) = SOneTo(8)

faceshape(::Hex27) = Quad9()
faces(::Hex27) = SVector(SVector(1,5,8,4,11,18,16,10,23), SVector(6,2,3,7,13,12,15,19,24), SVector(1,2,6,5,9,13,17,11,22), SVector(3,4,8,7,14,16,20,15,25), SVector(2,1,4,3,9,10,14,12,21), SVector(5,6,7,8,17,19,20,18,26))

function localnodes(::Type{T}, ::Hex27) where {T}
    SVector{27, Vec{3, T}}(
        (-1.0, -1.0, -1.0),
        ( 1.0, -1.0, -1.0),
        ( 1.0,  1.0, -1.0),
        (-1.0,  1.0, -1.0),
        (-1.0, -1.0,  1.0),
        ( 1.0, -1.0,  1.0),
        ( 1.0,  1.0,  1.0),
        (-1.0,  1.0,  1.0),
        ( 0.0, -1.0, -1.0),
        (-1.0,  0.0, -1.0),
        (-1.0, -1.0,  0.0),
        ( 1.0,  0.0, -1.0),
        ( 1.0, -1.0,  0.0),
        ( 0.0,  1.0, -1.0),
        ( 1.0,  1.0,  0.0),
        (-1.0,  1.0,  0.0),
        ( 0.0, -1.0,  1.0),
        (-1.0,  0.0,  1.0),
        ( 1.0,  0.0,  1.0),
        ( 0.0,  1.0,  1.0),
        ( 0.0,  0.0, -1.0),
        ( 0.0, -1.0,  0.0),
        (-1.0,  0.0,  0.0),
        ( 1.0,  0.0,  0.0),
        ( 0.0,  1.0,  0.0),
        ( 0.0,  0.0,  1.0),
        ( 0.0,  0.0,  0.0),
    )
end

function value(::Hex27, X::Vec{3, T}) where {T}
    ξ, η, ζ = X
    SVector{27, T}(
        -0.125 * ξ*η*ζ * (1-ξ) * (1-η) * (1-ζ),
         0.125 * ξ*η*ζ * (1+ξ) * (1-η) * (1-ζ),
        -0.125 * ξ*η*ζ * (1+ξ) * (1+η) * (1-ζ),
         0.125 * ξ*η*ζ * (1-ξ) * (1+η) * (1-ζ),
         0.125 * ξ*η*ζ * (1-ξ) * (1-η) * (1+ζ),
        -0.125 * ξ*η*ζ * (1+ξ) * (1-η) * (1+ζ),
         0.125 * ξ*η*ζ * (1+ξ) * (1+η) * (1+ζ),
        -0.125 * ξ*η*ζ * (1-ξ) * (1+η) * (1+ζ),
         0.25 * η*ζ * (1-ξ^2) * (1-η) * (1-ζ),
         0.25 * ξ*ζ * (1-ξ) * (1-η^2) * (1-ζ),
         0.25 * ξ*η * (1-ξ) * (1-η) * (1-ζ^2),
        -0.25 * ξ*ζ * (1+ξ) * (1-η^2) * (1-ζ),
        -0.25 * ξ*η * (1+ξ) * (1-η) * (1-ζ^2),
        -0.25 * η*ζ * (1-ξ^2) * (1+η) * (1-ζ),
         0.25 * ξ*η * (1+ξ) * (1+η) * (1-ζ^2),
        -0.25 * ξ*η * (1-ξ) * (1+η) * (1-ζ^2),
        -0.25 * η*ζ * (1-ξ^2) * (1-η) * (1+ζ),
        -0.25 * ξ*ζ * (1-ξ) * (1-η^2) * (1+ζ),
         0.25 * ξ*ζ * (1+ξ) * (1-η^2) * (1+ζ),
         0.25 * η*ζ * (1-ξ^2) * (1+η) * (1+ζ),
        -0.5 * ζ * (1-ξ^2) * (1-η^2) * (1-ζ),
        -0.5 * η * (1-ξ^2) * (1-η) * (1-ζ^2),
        -0.5 * ξ * (1-ξ) * (1-η^2) * (1-ζ^2),
         0.5 * ξ * (1+ξ) * (1-η^2) * (1-ζ^2),
         0.5 * η * (1-ξ^2) * (1+η) * (1-ζ^2),
         0.5 * ζ * (1-ξ^2) * (1-η^2) * (1+ζ),
         (1-ξ^2) * (1-η^2) * (1-ζ^2),
    )
end

# ---- Tri ----

abstract type Tri <: Shape{2} end

@doc raw"""
    Tri3()

# Geometry
```
η
^
|
3
|`\
|  `\
|    `\
|      `\
|        `\
1----------2 --> ξ
```
"""
struct Tri3 <: Tri end

get_order(::Tri3) = Order(1)
primarynodes_indices(::Tri3) = SOneTo(3)

faceshape(::Tri3) = Line2()
faces(::Tri3) = SVector(SVector(1,2), SVector(2,3), SVector(3,1))

function localnodes(::Type{T}, ::Tri3) where {T}
    SVector{3, Vec{2, T}}(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
    )
end

function value(::Tri3, X::Vec{2, T}) where {T}
    ξ, η = X
    SVector{3, T}(1-ξ-η, ξ, η)
end

@doc raw"""
    Tri6()

# Geometry
```
η
^
|
3
|`\
|  `\
5    `6
|      `\
|        `\
1-----4----2 --> ξ
```
"""
struct Tri6 <: Tri end

get_order(::Tri6) = Order(2)
primarynodes_indices(::Tri6) = SOneTo(3)

faceshape(::Tri6) = Line3()
faces(::Tri6) = SVector(SVector(1,2,4), SVector(2,3,6), SVector(3,1,5))

function localnodes(::Type{T}, ::Tri6) where {T}
    SVector{6, Vec{2, T}}(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.5),
    )
end

function value(::Tri6, X::Vec{2, T}) where {T}
    ξ, η = X
    ζ = 1 - ξ - η
    SVector{6, T}(
        ζ * (2ζ - 1),
        ξ * (2ξ - 1),
        η * (2η - 1),
        4ζ * ξ,
        4ζ * η,
        4ξ * η,
    )
end

# ---- Tet ----

abstract type Tet <: Shape{3} end

@doc raw"""
    Tet4()

# Geometry
```
                   η
                 .
               ,/
              /
           3
         ,/|`\
       ,/  |  `\
     ,/    '.   `\
   ,/       |     `\
 ,/         |       `\
1-----------'.--------2 --> ξ
 `\.         |      ,/
    `\.      |    ,/
       `\.   '. ,/
          `\. |/
             `4
                `\.
                   ` ζ
```
"""
struct Tet4 <: Tet end

get_order(::Tet4) = Order(1)
primarynodes_indices(::Tet4) = SOneTo(4)

faceshape(::Tet4) = Tri3()
faces(::Tet4) = SVector(SVector(2,1,3), SVector(1,4,3), SVector(4,2,3), SVector(1,2,4))

function localnodes(::Type{T}, ::Tet4) where {T}
    SVector{4, Vec{3, T}}(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
end

function value(::Tet4, X::Vec{3, T}) where {T}
    ξ, η, ζ = X
    χ = 1 - ξ - η - ζ
    SVector{4, T}(χ, ξ, η, ζ)
end

@doc raw"""
    Tet10()

# Geometry
```
                   η
                 .
               ,/
              /
           3
         ,/|`\
       ,/  |  `\
     ,6    '.   `8
   ,/       9     `\
 ,/         |       `\
1--------5--'.--------2 --> ξ
 `\.         |      ,/
    `\.      |    ,10
       `7.   '. ,/
          `\. |/
             `4
                `\.
                   ` ζ
```
"""
struct Tet10 <: Tet end

get_order(::Tet10) = Order(2)
primarynodes_indices(::Tet10) = SOneTo(4)

faceshape(::Tet10) = Tri6()
faces(::Tet10) = SVector(SVector(2,1,3,5,8,6), SVector(1,4,3,7,6,9), SVector(4,2,3,10,9,8), SVector(1,2,4,5,7,10))

function localnodes(::Type{T}, ::Tet10) where {T}
    SVector{10, Vec{3, T}}(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.5),
        (0.5, 0.0, 0.5),
    )
end

function value(::Tet10, X::Vec{3, T}) where {T}
    ξ, η, ζ = X
    χ = 1 - ξ - η - ζ
    SVector{10, T}(
        χ * (2χ - 1),
        ξ * (2ξ - 1),
        η * (2η - 1),
        ζ * (2ζ - 1),
        4ξ * χ,
        4η * χ,
        4ζ * χ,
        4ξ * η,
        4η * ζ,
        4ξ * ζ,
    )
end
