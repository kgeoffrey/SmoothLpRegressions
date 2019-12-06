### some regression stuff with Julia
using StatsBase
using Distributions
using Plots
using ForwardDiff

### Data ###
X = rand(500)*500
Y = X .^abs.(rand(Normal(1.,0.2), 500)) .+ rand(Normal(0,1), 500)

scatter(X,Y, ylims = (0,maximum(Y)), xlims = (0,500), markersize = 2)

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
f(x) = x'*w

plot!(f, X, label="Least Squares (L2)", linewidth = 2)

###### L0 norm approximation ###### best for ordinal and nominal data
## https://hal.archives-ouvertes.fr/hal-00173357/document


G_loss(X, Y, w, vari) = length(Y) - sum(exp.(-(X*w - Y).^2 ./ (2*vari)))
ww = rand(1)[1]
G_loss(X,Y, ww, var(X*w -Y))

closure = x -> G_loss(X, Y, x, 50)
grad(x) = ForwardDiff.derivative(closure, x)



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
g(X) = X*L0w
plot!(g, X, label = "L_0 Norm", linewidth = 2)




############ Linf Regression #############

linf(X, Y, w, L) = 2*L + log(sum(exp.(Y - X*w .- L)) + sum(exp.(X*w - Y .- L)))

l = maximum(abs.(X*w -Y))
linf(X, Y, w, l)
maximum((X*w - Y))

using ForwardDiff
clos = w -> linf(X, Y, w, l)
gradi(x) = ForwardDiff.gradient(clos, x)



function Linf_this(stepsize, epochs, X, Y)
    w = rand(size(X,2))[1]
    loss = []
    for i in 1:epochs
        l = maximum(abs.(X*w - Y))
        clos = w -> linf(X, Y, w, l)
        gradi(x) = ForwardDiff.derivative(clos, x)
        w = w - stepsize * gradi(w)
        append!(loss, clos(w))
    end
    return loss, w
end

lossinf, winf = Linf_this(0.001, 100, X, Y)

ginf(x) = x*winf
plot!(ginf, X, label = "L_inf Norm", linewidth = 2)

####### L1 norm approximation ########


smoothL1(X, Y, w, alph) = 1/alph * sum(log.(1 .+ exp.(-alph .* (X*w-Y))) + log.(1 .+ exp.(alph .* (X*w-Y))))


@time smoothL1(pp, 100)

pp = rand(4)


function L1_this(stepsize, epochs, X, Y)
    w = rand(size(X,2))[1]
    loss = []
    l = 1
    for i in 1:epochs
        l = (X*w - Y) .* l
        clos = w -> smoothL1(X, Y, w, l)
        gradi(x) = ForwardDiff.derivative(clos, x)
        w = w - stepsize * gradi(w)
        append!(loss, clos(w))
    end
    return loss, w
end

pp * 0.01
