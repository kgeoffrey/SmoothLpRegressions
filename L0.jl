### weird L0 regression
using HigherOrderDerivatives
using ForwardDiff

function addbias(X)
    X = hcat(ones(size(X, 1)), X)
    return X
end

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
    X = addbias(X)
    w = rand(size(X,2))
    loss = []
    variances= []
    for i in 1:epochs
        mw, variance = descent(stepsize, epochs/10, w, X, Y)
        w = mw
        append!(loss, G_loss(X, Y, w, variance))
        append!(variances, variance)
    end
    return loss, w, variances
end

XX = rand(100, 1)*100
YY = rand(100)*100

l0loss, w0,  variances = L0_this(0.0001, 1000, XX, YY)

plot(l0loss)


f(x) = w0[1] + x'*w0[2]
scatter(XX,YY)
plot!(f, XX, linewidth = 3, label = "L0 regression")
