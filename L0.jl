### weird L0 regression
using HigherOrderDerivatives
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


G_loss(X, Y, w, vari) = length(Y) - sum(exp.(-(X*w .- Y).^2 ./ (2*vari)))
closure = x -> G_loss(XX, YY, x, 0.5)

fff(x) = exp(x'*x)

derivative(fff, ww)







ForwardDiff.gradient(closure, ww)

t = w -> sum(-(XX*w .- YY).^2 )

gradient(closure, ww, 1)

t = Dual(ww)

exp.((t'*t)).g
