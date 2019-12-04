### weird L0 regression

function minimize_this(stepsize, epochs, w, X, Y)
    variance = var(X*w .- Y)
    closure = x -> G_loss(X, Y, x, variance)
    grad(x) = ForwardDiff.gradient(closure, x)

    for i in 1:epochs
        w = w - stepsize*grad(w)
    end
    return w, variance
end
