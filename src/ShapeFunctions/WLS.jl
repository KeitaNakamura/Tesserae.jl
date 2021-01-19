struct WLS{order, weight_order, dim} <: ShapeFunction{dim}
    poly::Polynomial{order}
    bspline::BSpline{weight_order, dim}
end

WLS{order}(bspline::BSpline) where {order} = WLS(Polynomial{order}(), bspline)

polynomial(wls::WLS) = wls.poly
weight_function(wls::WLS) = wls.bspline

support_length(wls::WLS) = support_length(weight_function(wls))
