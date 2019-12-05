## L infinity norm

using ForwardDiff

function descent(stepsize, epochs, w, X, Y)
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

XX = rand(100, 5)*100
YY = rand(100)*100
ww = rand(5)


### Linfinity norm

linf(X, Y, w, L) = 2*L + log(sum(exp.(Y - X*w .- L)) + sum(exp.(X*w - Y .- L)))

l = maximum(abs.(XX*ww -YY))
linf(XX, YY, ww, l)
maximum((XX*ww - YY))

using ForwardDiff
clos = w -> linf(XX, YY, w, l)
grad(x) = ForwardDiff.gradient(w -> linf(XX, YY, w, l), ww)


function Linf_this(stepsize, epochs, X, Y)
    w = rand(size(X,2))
    loss = []
    for i in 1:epochs
        l = maximum(abs.(X*w -Y))
        clos = w -> linf(XX, YY, w, l)
        grad(x) = ForwardDiff.gradient(clos, ww)
        w = w - stepsize*grad(w)
        append!(loss, clos(w))
    end
    return loss
end

lossinf = Linf_this(0.001, 100, XX, YY)
using Plots
plot(lossinf)
