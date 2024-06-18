# ft_dslr
---
## Usage

---
## Train model
### Use the gradient descent
The logistic regression is a way to construct a model able to give a probability to be in a certain state
(0 or 1, happy or sad, etc...).

In this case the sigmoid function seams to be a great tool to describe what we are looking for.

Let define the sigmoid as :
```math
    \sigma(x) = {1 \over {1 + e^{-(\beta_0 + \beta_1 * x)}}}
```
or
```math
    \sigma(x) = {1 \over {1 + e^{-z(x)}}}
```
where
```math
    z(x) = \beta_0 + \beta_1 * x
```

To find the best param $$\beta_0$$ and $$\beta_1$$ we search to minimise the cost function associate
to the sigmoid.

Bernoulli distribution:
```math
    P(y = Y) = \sigma(x)^{y} * (1 - \sigma(x))^{(1 - y)}
```
