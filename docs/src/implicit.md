# Implicit formulation

## Residual vector and Jacobian matrix

In MPM, since the momentum equation is solved on the grid, the residual vector $\bm{R}$ for solving nonlinear equations can be expressed as:

```math
\bm{R}_I(\bm{u}_I^{n+1}) = m_I \bm{a}_I(\bm{u}_I^{n+1}) + \bm{f}_I(\bm{u}_I^{n+1}),
```

where

```math
\bm{f}_I
= \sum_p V_p \bm{\sigma}_p \frac{\partial w_{Ip}}{\partial \bm{x}}
= \sum_p V_p^0 \bm{\tau}_p \frac{\partial w_{Ip}}{\partial \bm{x}}.
```

Using the Newmark-$\beta$ method, the grid velocity and acceleration can be written as:

```math
\begin{aligned}
\bm{v}_I^{n+1} &= \frac{\gamma}{\beta\Delta{t}} \bm{u}_I^{n+1} - \left( \frac{\gamma}{\beta} - 1 \right) \bm{v}_I^n - \Delta{t} \left( \frac{\gamma}{2\beta} - 1\right) \bm{a}_I^n, \\
\bm{a}_I^{n+1} &= \frac{1}{\beta\Delta{t}^2} \bm{u}_I^{n+1} - \frac{1}{\beta\Delta{t}} \bm{v}_I^n - \left( \frac{1}{2\beta} -1 \right) \bm{a}_I^n.
\end{aligned}
```

The linearization of the residual vector is given by:

```math
\delta\bm{R}_I = m_I \frac{\partial\bm{a}_I^{n+1}}{\partial\bm{u}_J^{n+1}} \delta\bm{u}_J + \frac{\partial\bm{f}_I^{n+1}}{\partial\bm{u}_J^{n+1}} \delta\bm{u}_J,
```

where

```math
\begin{aligned}
\left( \frac{\partial\bm{a}_I^{n+1}}{\partial\bm{u}_J^{n+1}} \right)_{ij} &= \frac{\delta_{IJ}}{\beta\Delta{t}^2} \delta_{ij}, \\
\left( \frac{\partial\bm{f}_I^{n+1}}{\partial\bm{u}_J^{n+1}} \right)_{ij} &= \sum_p \frac{\partial w_{Ip}}{\partial x_k} \left(\mathbb{C}_p\right)_{ikjl} \frac{\partial w_{Jp}}{\partial x_l} V_p^0
\end{aligned}
```

The spatial tangent modulus $\mathbb{C}$ is defined as:

```math
\mathbb{C}_{ijkl} = \frac{\partial \tau_{ij}}{\partial F_{km}} F_{lm} - \tau_{il}\delta_{jk}
```

## The derivative of the basis function[^1]

In the implicit formulation, the derivative of the basis function $\nabla_{\bm{x}}w$ must be evaluated based on the current configuration.
However, in MPM, this is not feasible because the basis function and its derivative are defined on the coordinates at the beginning of each time step, $\bm{X}$.
Thus the derivative of the basis function with respect to the current coordinates $\bm{x}$ should be computed using

[^1]: [Charlton, T.J., Coombs, W.M. and Augarde, C.E., 2017. iGIMP: An implicit generalised interpolation material point method for large deformations. Computers & Structures, 190, pp.108-125.](https://doi.org/10.1016/j.compstruc.2017.05.004)

```math
\frac{\partial w}{\partial x_j}
= \frac{\partial w}{\partial X_i} \frac{\partial X_i}{\partial x_j}
= \frac{\partial w}{\partial X_i} \Delta{F}_{ij}^{-1}
```

where

```math
\Delta{F}_{ij}
= \frac{\partial x_i}{\partial X_j}
= \delta_{ij} + \frac{\partial u_i}{\partial X_j}.
```

This relative deformation gradient $\Delta\bm{F}$ is typically used to update the deformation gradient in MPM:

```math
F_{ij} = \Delta{F}_{ik} F_{kj}^n.
```
