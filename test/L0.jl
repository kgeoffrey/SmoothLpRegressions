### weird L0 regression
using HigherOrderDerivatives
using ForwardDiff

function addbias(X)
    X = hcat(ones(size(X, 1)), X)
    return X
end

G_loss(X, Y, w, vari) = length(Y) - sum(exp.(-(X*w - Y).^2 ./ (2*vari)))

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


XX = collect(1:100)
YY = XX .^(2 *(rand(Normal(1.,0), 1))) .+ rand(Normal(10,1000), 100)
newx = addsquare(XX)


l0loss, w0, variances = L0_this(0.01, 100, newx, YY)

plot(l0loss)



scatter(XX,YY)
plot!(f, XX, linewidth = 3, label = "L0 regression")




function addsquare(X)
    X = hcat(X, X.^2)
    return X
end

newx = addsquare(XX)
lossinf, winf = Linf_this(0.00001, 1000, newx, YY)

ff(x) = w0[1] .+ x[:,1] .*w0[2] .+ x[:,2] .*w0[3]

mm = ff(newx)

plot!(mm)
