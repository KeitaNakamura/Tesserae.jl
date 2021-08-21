@generated function initval(::Type{T}) where {T}
    if Base._return_type(zero, (T,)) == Union{}
        exps = [:(zero($t)) for t in fieldtypes(T)]
        quote
            @_inline_meta
            T($(exps...))
        end
    else
        quote
            @_inline_meta
            zero(T)
        end
    end
end
@generated function initval(::Type{T}) where {T <: NamedTuple}
    exps = [:(zero($t)) for t in fieldtypes(T)]
    quote
        @_inline_meta
        T(($(exps...),))
    end
end
initval(x) = initval(typeof(x))

reinit!(x::AbstractArray) = (broadcast!(initval, x, x); x)


@generated function interpolate(x::T, y::T, α::Real) where {T}
    if Base._return_type(zero, (T,)) == Union{}
        exps = [:(α*x.$name + (1-α)*y.$name) for name in fieldnames(T)]
        quote
            @_inline_meta
            T($(exps...))
        end
    else
        quote
            @_inline_meta
            α*x + (1-α)*y
        end
    end
end
@generated function interpolate(x::T, y::T, α::Real) where {T <: NamedTuple}
    exps = [:(α*x.$name + (1-α)*y.$name) for name in fieldnames(T)]
    quote
        @_inline_meta
        T(($(exps...),))
    end
end


function Tensor3D(x::SecondOrderTensor{2,T}) where {T}
    z = zero(T)
    @inbounds SecondOrderTensor{3,T}(x[1,1], x[2,1], z, x[1,2], x[2,2], z, z, z, z)
end

function Tensor3D(x::SymmetricSecondOrderTensor{2,T}) where {T}
    z = zero(T)
    @inbounds SymmetricSecondOrderTensor{3,T}(x[1,1], x[2,1], z, x[2,2], z, z)
end

function Tensor2D(x::SecondOrderTensor{3,T}) where {T}
    @inbounds SecondOrderTensor{2,T}(x[1,1], x[2,1], x[2,1], x[2,2])
end

function Tensor2D(x::SymmetricSecondOrderTensor{3,T}) where {T}
    @inbounds SymmetricSecondOrderTensor{2,T}(x[1,1], x[2,1], x[2,2])
end

function Tensor2D(x::FourthOrderTensor{3,T}) where {T}
    @inbounds FourthOrderTensor{2,T}((i,j,k,l) -> @inbounds(x[i,j,k,l]))
end

function Tensor2D(x::SymmetricFourthOrderTensor{3,T}) where {T}
    @inbounds SymmetricFourthOrderTensor{2,T}((i,j,k,l) -> @inbounds(x[i,j,k,l]))
end
