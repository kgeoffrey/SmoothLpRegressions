## L infinity norm

using ForwardDiff
using Plots

function addbias(X)
    X = hcat(ones(size(X, 1)), X)
    return X
end



### Linfinity norm

linf(X, Y, w, L) = L + log(sum(exp.(Y - X*w .- L)) + sum(exp.(X*w - Y .- L)))


function Linf_this(stepsize, epochs, X, Y)
    X = addbias(X)
    w = rand(size(X,2))
    loss = []
    for i in 1:epochs
        l = maximum(abs.(X*w -Y))
        clos = w -> linf(X, Y, w, l)
        grad(x) = ForwardDiff.gradient(clos, x)
        w = w - stepsize*grad(w)
        append!(loss, clos(w))
    end
    return loss, w
end


###testing

XX = rand(500)*500
YY = XX .^abs.(rand(Normal(1.,0.2), 500)) .+ rand(Normal(0,1), 500)

lossinf, winf = Linf_this(0.00001, 10000, XX, YY)


f(x) = winf[1] + x'*winf[2]
scatter(XX, YY)
plot!(f, XX)
