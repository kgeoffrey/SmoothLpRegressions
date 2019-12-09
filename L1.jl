### L1 regression

## smooth L1 regression: http://pages.cs.wisc.edu/~gfung/GeneralL1/L1_approx_bounds.pdf


smoothL1(X, Y, w, alph) = 1/alph * sum(log.(1 .+ exp.(-alph .* (X*w-Y))) + log.(1 .+ exp.(alph .* (X*w-Y))))

smoothL1(XX,YY, 0.99, 0.001)




@time smoothL1(pp, 100)

pp = rand(4)

function L1_minimize(stepsize, w, gradi, epochs)
    for i in 1:epochs
        w = w - stepsize * gradi(w)
    end
    return w
end


function L1_this(stepsize, epochs, X, Y)
    X = addbias(X)
    w = rand(size(X,2))
    loss = []
    ls = []
    for i in 1:epochs
        l = 1/(2 *(var(X*w - Y)))
        # l = 13 / maximum(abs.(X*w - Y)) another rule!
        clos = x -> smoothL1(X, Y, x, l)
        gradi(x) = ForwardDiff.gradient(clos, x)
        w = L1_minimize(stepsize, w, gradi, epochs/10)
        append!(loss, clos(w))
        append!(ls, l)
    end
    return loss, w, ls
end

l1loss, l1w, ls = L1_this(0.001, 100, XX, YY)

plot(ls)




exp(l * (X*w-Y)) = 1


f(x) = l1w[1] + x'*l1w[2]

scatter(XX, YY)
plot!(f, XX, linewidth = 3, label = "L1 regression")
