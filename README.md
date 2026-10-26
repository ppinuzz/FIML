# Field Inversion and Machine Learning (FIML) #

Create the `FIML` enviroment:
```
mamba create --name FIML python=3.13.2 numpy=2.2.4 matplotlib=3.10.1 prettytable=3.16.0 scipy=1.15.2 colorama=0.4.6 pyyaml=6.0.2 pandas=3.0.6 spyder-kernels=3.0 h5py=3.16.0
```
Install JAX separately with
```
pip install -U jax
```

## $\beta$ model available
- `BxBt`: space- and time-only basis functions
```math
\beta(x,t) = \beta_\mathrm{ref} + \sum_{m=1}^M \sum_{q=1}^Q B_x^m(x) \theta_{mq} B_t^q(t) = \beta_\mathrm{ref} + z(x,t) 
```
$\beta$ can be negative

- `BxBt_exp`: exponentiated version of `BxBt`
```math
\beta(x,t) = e^z
```

- `BxBt_softplus`: smoothened version of `BxBt` using the softplus function $S(z) = \ln(1 + z)$
```math
\beta(x,t) = \beta_\mathrm{min} + (1-\beta_\mathrm{min}) \frac{S(z)}{\ln(2)}
```
where $\beta_\mathrm{min} > 0$ is a minimum value that can be imposed

- `RBF_gauss`: Gaussian RBFs (for steady problems)
```math
\beta(x) = \sum_{k=1}^W \phi_k(x) w_k = \sum_{k=1}^W e^{-K(x-c_k)^2} w_k
```