### run test

include("regressions.jl")

function randdata(n)
    X = abs.(rand(n)*n)
    Y = X .^((rand(Beta(50,1.8), n))) .+ (rand(Normal(1,50), n))
    return X, Y
end

XX, YY = randdata(1000)
scatter(XX, YY, markersize = 1)

linfmodel = fit(Linf, XX, YY, 100, 0.0001)
olsmodel = fit(OLS, XX, YY)
l1model = fit(L1, XX, YY, 100, 0.000001)
l0model = fit(L0, XX, YY, 100, 0.00001)


f(x) = linfmodel.w[1] + x'*linfmodel.w[2]
g(x) = olsmodel.w[1] + x'*olsmodel.w[2]
h(x) = l1model.w[1] + x'*l1model.w[2]
j(x) = l0model.w[1] + x'*l0model.w[2]

scatter(XX, YY, markersize = 1)
plot!(f, XX, linewidth = 3, label = "LINF")
plot!(g, XX, linewidth = 3, label = "OLS")
plot!(h, XX, linewidth = 3, label = "L1")
plot!(j, XX, linewidth = 3, label = "L0")
