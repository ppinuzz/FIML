# Burgers' equation

## Directory structure
- `problem_setup`: setup file used to load BCs, ICs, discretisation steps and $\nu$ function for each Burgers' problem
- `solution_Burgers`: solution files created by Burgers' solver


## True model: variable viscosity
$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \frac{\partial}{\partial x} \bigg(\nu(x,t) \, \frac{\partial u}{\partial x} \bigg)
$$ 
with $x \in [0, 1]$ and $t \in [0, T]$

## Deficient model: constant viscosity
$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} = \nu_0 \frac{\partial^2 u}{\partial x^2}
$$ 
with $x \in [0, 1]$ and $t \in [0, T]$