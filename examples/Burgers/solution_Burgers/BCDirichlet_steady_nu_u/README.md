# `BCDirichlet_steady_nu_u`

## Problem
Steady-state 1D Burgers' equation (in non-dimensional form)
```math
u \frac{du}{dx} = \frac{d}{dx} \bigg( \nu \frac{du}{dx} \bigg)
```
on the space domain $(0, 1)$, with Dirichlet boundary conditions
```math
u(0) = 0
u(1) = 1
```
and variable viscosity
```math
\nu = \nu(u) = \nu_0 (1 + 20u^2 + 2 \sin(u)) = \nu_0 \beta(u)
```

## Data files
- `solution.h5`: solution on an extremely fine mesh ($\dx = 10^{-5}$) of the exact equation ("exact solution", not considering numerical errors)
- `probes_<N>.h5`: exact solution sampled at $N$ equally-spaced points (artificial "experimental data")