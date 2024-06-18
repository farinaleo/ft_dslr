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
    P(Y = y) = \sigma(x)^{y} * (1 - \sigma(x))^{(1 - y)}
```
Cost Function
```math
    L = \prod^{n - 1}_{i = 0} P(Y = y)
```
```math
    L = \prod^{n - 1}_{i = 0} \sigma(x)^{y} * (1 - \sigma(x))^{(1 - y)}
```
To counteract the fact that a product of numbers between 0 and 1 tends towards 0,
we apply the log function to each side of the equation.
```math
    log(L) = log(\prod^{n - 1}_{i = 0} \sigma(x)^{y} * (1 - \sigma(x))^{(1 - y)})
```
```math
    log(L) = \sum^{n - 1}_{i = 0} log(\sigma(x)^{y} * (1 - \sigma(x))^{(1 - y)})
```
```math
    log(L) = \sum^{n - 1}_{i = 0} log(\sigma(x)^{y}) + log((1 - \sigma(x))^{(1 - y)})
```
```math
    log(L) = \sum^{n - 1}_{i = 0} log(\sigma(x)^{y}) + (1 - y)log((1 - \sigma(x)))
```