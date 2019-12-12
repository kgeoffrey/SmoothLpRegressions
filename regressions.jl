### Regressions Test
using HigherOrderDerivatives
using ForwardDiff
using Distributions
using Plots

abstract type Regression end

mutable struct Linf <: Regression
    w::AbstractArray
    c::Real
    loss::AbstractArray
end

mutable struct OLS <: Regression
    w::AbstractArray
    c::Real
    loss::AbstractArray
end

mutable struct L1 <: Regression
    w::AbstractArray
    c::Real
    loss::AbstractArray
end

mutable struct L0 <: Regression
    w::AbstractArray
    c::Real
    loss::AbstractArray
end


function addbias(X::AbstractArray)
    X = hcat(ones(size(X, 1)), X)
    return X
end

function initialize(obj::Type{<:Regression}, X)
    w = rand(size(X,2))
    c = 0
    loss = []
    return obj(w, c, loss)
end

function minimize!(obj::Regression, grad::Function, epochs::Int, stepsize::Real)
    for i in 1:epochs
        obj.w = obj.w - stepsize*grad(obj.w)
    end
end

function metrics!(obj::Regression, f::Function)
    append!(obj.loss, f(obj.w))
end

Linf_loss(obj::Regression, X, Y, w) = obj.c + log(sum(exp.(Y - X*w .- obj.c)) + sum(exp.(X*w - Y .- obj.c)))
L1_loss(obj::Regression, X, Y, w) = 1/obj.c * sum(log.(1 .+ exp.(-obj.c .* (X*w-Y))) + log.(1 .+ exp.(obj.c .* (X*w-Y))))
L0_loss(obj::Regression, X, Y, w) = length(Y) - sum(exp.(-(X*w - Y).^2 ./ (2*obj.c)))

function fit(::Type{Linf}, X::AbstractArray, Y::AbstractArray, epochs::Int, stepsize::Real)
    X = addbias(X)
    M = initialize(Linf, X)
    loss = []
    for i in 1:epochs
        M.c = maximum(abs.(X*M.w -Y))
        close = x -> Linf_loss(M, X, Y, x)
        grad(x) = gradient(close, x)

        minimize!(M, grad, 1, stepsize)
        metrics!(M, close)
    end
    return M
end

function fit(::Type{OLS}, X::AbstractArray, Y::AbstractArray)
    X = addbias(X)
    M = initialize(OLS, X)
    M.w = inv(X'*X)*X'*Y
    return M
end

function fit(::Type{L1}, X::AbstractArray, Y::AbstractArray, epochs::Int, stepsize::Real)
    X = addbias(X)
    M = initialize(L1, X)
    loss = []
    for i in 1:epochs
        M.c = 1/(2 *(var(X*M.w - Y)))
        close = w -> L1_loss(M, X, Y, w)
        grad(x) = gradient(close, x)

        minimize!(M, grad, Int(floor(epochs/10)), stepsize)
        metrics!(M, close)
    end
    return M
end

function fit(::Type{L0}, X::AbstractArray, Y::AbstractArray, epochs::Int, stepsize::Real)
    X = addbias(X)
    M = initialize(L0, X)
    loss = []
    for i in 1:epochs
        M.c = var(X*M.w .- Y)
        close = w -> L0_loss(M, X, Y, w)
        grad(x) = gradient(close, x)

        minimize!(M, grad, Int(floor(epochs/10)), stepsize)
        metrics!(M, close)
    end
    return M
end
