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
\beta(x,t) = \beta_\mathrm{ref} + \sum_{m=1}^M \sum_{q=1}^Q B_x^m(x) \theta_{mq} B_t^q(t) = \beta_\mathrm{ref} + z(x,t) \gtreqless 0
```

- `BxBt_exp`: exponentiated version of `BxBt`
```math
\beta(x,t) = e^z > 0
```

- `BxBt_softplus`: smoothened version of `BxBt` using the softplus function $s(z) = \ln(1 + e^z)$
```math
\beta(x,t) = \beta_\mathrm{min} + (1-\beta_\mathrm{min}) \frac{S(z)}{\ln(2)} = G(z; \beta_\mathrm{min}) \ge \beta_\mathrm{min}
```
where $0 \le \beta_\mathrm{min} 1 1$ is a minimum value that can be imposed. With this model, $\beta(z) > 0$ always, $\theta = 0$ is the baseline $\beta = 1$, and $\beta$ does not increase as quickly as the `BxBt_exp` model

- `RBF_Gauss_x`: Gaussian RBFs (for steady problems)
```math
\beta(x) = \beta_\mathrm{ref} + \sum_{k=1}^W \phi_k(x) w_k = \sum_{k=1}^W \exp(-K(x-c_k)^2) w_k \gtreqless 0
```

- `RBF_Gauss_u`: same, but depending on the solution $u$, not on the independent coordinate $x$
```math
\beta(u) = \beta_\mathrm{ref} + \sum_{k=1}^W \phi_k(u) w_k = \sum_{k=1}^W \exp(-K(u-u_{\mathrm{c},k})^2) w_k \gtreqless 0
```
the centre of each RBF is the solution at a fixed space position: $u_{\mathrm{c},k} = u(x_k)$

- `RBF_Gauss_x_softplus`: same as `RBF_Gauss_x`, but the positivity-preserving function $G(z; \beta_\mathrm{min})$ is applied to the $\beta$ obtained by `RBF_Gauss_x`
```math
\beta(x) = G\bigg(\sum_{k=1}^W \phi_k(x) w_k; \beta_\mathrm{min}\bigg) = G\bigg(\sum_{k=1}^W \exp(-K(x-c_k)^2) w_k; \beta_\mathrm{min}\bigg) > 0
```

- `RBF_Gauss_u_softplus`: same as `RBF_Gauss_u`, but the positivity-preserving function $G(z; \beta_\mathrm{min})$ is applied to the $\beta$ obtained by `RBF_Gauss_u`
```math
\beta(x) = G\bigg(\sum_{k=1}^W \phi_k(u) w_k; \beta_\mathrm{min} \bigg) = G\bigg(\sum_{k=1}^W \exp(-K(u-u_{\mathrm{c},k})^2) w_k; \beta_\mathrm{min} \bigg) > 0
```