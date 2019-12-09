### L1 regression

## smooth L1 regression: http://pages.cs.wisc.edu/~gfung/GeneralL1/L1_approx_bounds.pdf


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
