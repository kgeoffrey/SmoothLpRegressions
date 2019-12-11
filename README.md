# ExoticRegressions
In this repo I tried to implement some hacks around finding measures of central tendency (L0, L1, Linf) for regression problems.
In terms of LP spaces the central tendency function for mean is the L2-norm, for the median it is the L1-Norm. Arguably less known 
are other norms for regression problems: the L-Infinity-Norm that is an estimator for the Mid-Range and the L0-"Norm" (not really a norm),
that estimates the Mode.

There are some problems with some of the L-norms above (L0, L1, Linf) which you can see by looking at the norm ball:

<p align="center">
<img src="../master/img/pnorm.png" alt="drawing" width="400"/>
</p>

All the norms are convex, except the L0-Norm, which makes it very hard (in fact NP hard) to optimize. If you think of norms in terms of
distance, for the L0-norm, we are trying to minimize the number of nonzero elements of a vector. For the L-infinity-Norm we are only 
trying to minimize the maximum distance (absolute value) of a vector. For instance for a linear regression problem where the loss function 
is the L0-norm, we would try to find the coefficient of the equation of a line that goes directly through as many points as possible (if a
line goes directly through a data point the magnitude of the error is 0!).

In the context of regression, we are trying to minimize the loss functions (L0, L1, L2 and Linf-norms). Going back to the norm ball above,
you see that the L1 and L2-norms are not differentiable at every point and the L0 norm is not convex and not differentiable at every point.

Yet these norms can still be more or less well approximated! See the approximations below, everything that follows a sum is applied elementwise (sorry too lazy):

### L-Infinity-Norm
This norm can be approxmiated as follows:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cleft%20%5C%7C%20Xw%20-%20Y%20%5Cright%20%5C%7C_%7B%5Cinfty%7D%20%5Capprox%20%20log(%5Csum%20%5Cexp(Xw%20-%20Y)%20%2B%20%5Csum%20%5Cexp(Y%20-%20Xw))">
</p>

Source: https://www.cs.ubc.ca/~schmidtm/Courses/340-F15/L15.pdf

### L1-Norm
We can approximate the L1-norm by a differentiable function:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cleft%20%5C%7C%20Xw%20-%20Y%20%5Cright%20%5C%7C_%7B1%7D%20%5Capprox%20%5Cfrac%7B1%7D%7B%5Calpha%7D%20%5Csum%20log(1%20%2B%20exp(-%5Calpha%20(Xw%20-%20Y)))%20%2Blog(1%20%2B%20exp(%5Calpha%20(Xw%20-%20Y))">
</p>

Source: http://pages.cs.wisc.edu/~gfung/GeneralL1/L1_approx_bounds.pdf


### L0-Norm
This norm is a bit more tricky to represent as a differentiable function, the algorithm to minimize it is based on the "SL0 Algorithm".

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Cleft%20%5C%7C%20Xw%20-%20Y%20%5Cright%20%5C%7C_%7B0%7D%20%5Capprox%20n%20-%20%5Csum%20%5Cexp%20%5Cbigg(%5Cfrac%7B(Xw%20-%20Y)%5E2%7D%7B2%5Csigma%20%5E%7B2%7D%7D%20%5Cbigg)">
</p>

Source: http://pages.cs.wisc.edu/~gfung/GeneralL1/L1_approx_bounds.pdf

The approximation can now be minimized via gradient descent! To find the gradients I used another package of mine: HigherOrderDerivatives (add link). Note that for both L1 and L0 an additional parameter is required, for L0 I am computing the variance
of Xw - ) in each step and similarily I am using 1 /(2 * variance(Xw-Y)) for L1 regression. I tried making use of LogSumExp tricks to prevent over- and underflow, but convergence is not always guaranteed. Gradient descent with L0-norm regression may get stuck in local minima.

## Data

Here I have a super noisy data set. The L0-norm is least affected by outliers and the L-Infinity_Norm the most:

<p align="center">
<img src="../master/img/comparison.png" alt="drawing" width="600"/>
</p>
