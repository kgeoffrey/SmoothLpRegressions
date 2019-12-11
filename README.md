# ExoticRegressions
In this repo I tried to implement some hacks around finding measures of central tendency (L0, L1, Linf) for regression problems.
In terms of LP spaces the central tendency function for mean is the L2-norm, for the median it is the L1-Norm. Arguably less known 
are other norms for regression problems: the L-Infinity-Norm that is an estimator for the Mid-Range and the L0-"Norm" (not really a norm),
that estimates the Mode.

There are some problems with some of the L-norms above (L0, L1, Linf) which you can see by looking at the norm ball:


All the norms are convex, except the L0-Norm, which makes it very hard (in fact NP hard) to optimize. If you think of norms in terms of
distance, for the L0-norm, we are trying to minimize the number of nonzero elements of a vector. For the L-infinity-Norm we are only 
trying to minimize the maximum distance (absolute value) of a vector. For instance for a linear regression problem where the loss function 
is the L0-norm, we would try to find the coefficient of the equation of a line that goes directly through as many points as possible (if a
line goes directly through a data point the magnitude of the error is 0!).

In the context of regression, we are trying to minimize the loss functions (L0, L1, L2 and Linf-norms). Going back to the norm ball above,
you see that the L1 and L2-norms are not differentiable at every point and the L0 norm is not convex and not differentiable at every point.

Yet these norms can still be more or less well approximated! 






<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">
