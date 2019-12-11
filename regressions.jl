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




function randdata(n)
    # X = rand(1000)*1000
    # Y = X .^(rand(Normal(1.,0.1), 1000)) .+ rand(Normal(0,1), 1000)
    X = abs.(rand(n)*n)
    Y = X .^((rand(Beta(50,1.8), n))) .+ (rand(Normal(1,50), n))
    return X, Y
end

XX, YY = randdata(1000)
scatter(XX, YY, markersize = 1)

model = fit(Linf, XX, YY, 100, 0.0001)
plot(model.loss)
olsmodel = fit(OLS, XX, YY)
l1model = fit(L1, XX, YY, 100, 0.000001)
plot(l1model.loss)
l0model = fit(L0, XX, YY, 100, 0.00001)
plot(l0model.loss)


f(x) = model.w[1] + x'*model.w[2]
g(x) = olsmodel.w[1] + x'*olsmodel.w[2]
h(x) = l1model.w[1] + x'*l1model.w[2]
j(x) = l0model.w[1] + x'*l0model.w[2]

scatter(XX, YY, markersize = 1)
plot!(f, XX, linewidth = 3, label = "LINF")
plot!(g, XX, linewidth = 3, label = "OLS")
plot!(h, XX, linewidth = 3, label = "L1")
plot!(j, XX, linewidth = 3, label = "L0")
