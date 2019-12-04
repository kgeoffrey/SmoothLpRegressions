### some regression stuff with Julia
using StatsBase
using Distributions
using Plots
using ForwardDiff

### Data ###
X = rand(50)*50
Y = X .^rand(Normal(1.1,0.2), 50) .+ rand(Normal(0,3), 50)

scatter(X,Y, ylims = (0,maximum(Y)), xlims = (0,50))

mse_(x) = sum(x.^2)
function addbias(X)
    X = hcat(ones(length(X)), X)
    return X
end

function least_squares(X, Y)
    X = addbias(X)
    w = inv(X'*X)*X'*Y
    return w[1], w[2]
end

intercept, w = least_squares(X,Y)
f(x) = intercept .+ x'*w

plot!(f, X, label="Least Squares")

###### L0 norm approximation ######
## https://hal.archives-ouvertes.fr/hal-00173357/document


G_loss(X, Y, w, vari) = length(Y) - sum(exp.(-(X*w - Y).^2 ./ (2*vari)))
ww = rand(1)[1]
G_loss(X,Y, ww, var(X*w -Y))

closure = x -> G_loss(X, Y, x, vv)
grad(x) = ForwardDiff.derivative(closure, x)

tre(ww)

function minimize(stepsize, epochs, w)
    variance = var(X*w -Y)
    closure = x -> G_loss(X, Y, x, variance)
    grad(x) = ForwardDiff.derivative(closure, x)

    for i in 1:epochs
        w = w - stepsize*grad(w)
    end
    return w, variance
end

function L0_min(stepsize, epochs)
    w = rand(1)[1]
    loss = []
    variances= []
    for i in 1:epochs
        mw, variance = minimize(stepsize, epochs/10, w)
        w = mw
        append!(loss, closure(w))
        append!(variances, variance)
    end
    return loss, w, variances
end

lossL0, L0w, vars = L0_min(0.0001, 100)

plot!(g, X, label = "L0 Loss")

############ Multi Dimensional L0 Regression #############

XX = rand(100, 5)*100
YY = rand(100)*100
ww = rand(5)

G_loss(X, Y, w, vari) = length(Y) - sum(exp.(-(X*w - Y).^2 ./ (2*vari)))
closure = x -> G_loss(XX, YY, x, vv)
grad(x) = ForwardDiff.derivative(closure, x)

function minimize_this(stepsize, epochs, w, X, Y)
    variance = var(X*w .- Y)
    closure = x -> G_loss(X, Y, x, variance)
    grad(x) = ForwardDiff.gradient(closure, x)

    for i in 1:epochs
        w = w - stepsize*grad(w)
    end
    return w, variance
end

function L0_this(stepsize, epochs, X, Y)
    w = rand(size(X,2))
    loss = []
    variances= []
    for i in 1:epochs
        mw, variance = minimize_this(stepsize, epochs/10, w, X, Y)
        w = mw
        append!(loss, G_loss(X, Y, w, variance))
        append!(variances, variance)
    end
    return loss, w, variances
end

newloss, neww, sig2 = L0_this(0.0001, 100, XX, YY)

plot(newloss)

newloss

plot(sig2)
