abstract type ShapeFunction{dim} end

# scalar
(f::ShapeFunction{0})(ξ::Real, args...) where {dim} = value(f, ξ, args...)

function value_derivative(f::ShapeFunction, ξ::Real, args...)
    res = DiffResults.DiffResult(zero(ξ), zero(ξ))
    res = ForwardDiff.derivative!(res, ξ -> value(f, ξ, args...), ξ)
    DiffResults.value(res), DiffResults.derivative(res)
end

function derivative(f::ShapeFunction, ξ::Real, args...)
    value_derivative(f, ξ, args...)[2]
end

# vector
(f::ShapeFunction{dim})(ξ::Vec{dim}, args...) where {dim} = value(f, ξ, args...)

function value_gradient(f::ShapeFunction{dim}, ξ::Vec{dim}, args...) where {dim}
    dv, v = gradient(ξ -> value(f, ξ, args...), ξ, :all)
    v, dv
end

function gradient(f::ShapeFunction{dim}, ξ::Vec{dim}, args...) where {dim}
    value_gradient(f, ξ, args...)[2]
end


struct GradientShapeFunction{dim, F <: ShapeFunction{dim}} <: ShapeFunction{dim}
    f::F
end

Base.adjoint(f::ShapeFunction) = GradientShapeFunction(f)
TensorValues.∇(f::ShapeFunction) = GradientShapeFunction(f)

value(∇f::GradientShapeFunction{0}, ξ::Real, args...) = derivative(∇f.f, ξ, args...)
value(∇f::GradientShapeFunction{dim}, ξ::Vec{dim}, args...) where {dim} = gradient(∇f.f, ξ, args...)
