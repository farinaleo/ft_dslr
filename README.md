# ft_dslr
---
## Usage
### For the required data analysis:

_Move to the corresponding directory (in srcs)_
- Describe:
  - Run `python ft_dslr/describe/describe.py <file>`
- Histogram:
  - Run `python ft_dslr/histogram/histogram.py <file>`
- Scatter plot:
  - Run `python ft_dslr/scatter_plot/scatter_plot.py <file>`
- Pair plot:
  - Run `python ft_dslr/pair_plot/pair_plot.py <file>`

All script came with its associate jupyter notebook witch give us more information and answers for
the questions of the subject.

### For the required training and predict process:

Scripts `logreg_train.py` and `logreg_predict.py`, required by the subject, must be used to train a model
and predict the house of a student.

The accuracy of the model must greater or equal to 98%. It means that can only have a 2% of error during the
predict process.


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

To find the best param $\beta_0$ and $\beta_1$ we search to minimise the cost function associate
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
This gives us the following cost function, known as the "log loss" function.
```math
    \mathcal{L} = \sum^{n - 1}_{i = 0} y * log(\sigma(x)) + (1 - y) * log((1 - \sigma(x)))
```
To find the best parameters $\beta_0$ and $\beta_1$ we want to minimise the coset function with
a gradient descent.

For each iteration of the gradient descent we apply the following formula:
```math
    \beta_0 = \beta_{0(prev)} - \alpha * {\partial \mathcal{L} \over \partial \beta_0}
```
```math
    \beta_1 = \beta_{1(prev)} - \alpha * {\partial \mathcal{L} \over \partial \beta_1}
```
where
```math
    {\partial \mathcal{L} \over \partial \beta_0} = {\partial \mathcal{L} \over \partial \sigma}
     * {\partial \sigma \over \partial \beta_0}
     = \sum^{n - 1}_{i = 0} (\sigma(x^{(i)}) - y^{(i)}) * x^{(i)}
```
and
```math
    {\partial \mathcal{L} \over \partial \beta_1} = {\partial \mathcal{L} \over \partial \sigma}
     * {\partial \sigma \over \partial \beta_1}
     = \sum^{n - 1}_{i = 0} (\sigma(x^{(i)}) - y^{(i)})
```

code for the logical regression:
```python
import numpy as np

def gradient_descent(X: np.ndarray, y: np.ndarray, epoch:int, learning_rate:float)->dict:
    thetas = {
        'theta_0':0,
        'theta_1':1
    }
    
    for _ in range(epoch):
        _p0 = partial_derivative_theta_0(X, y)
        _p1 = partial_derivative_theta_1(X, y)
        thetas = {
            'theta_0':thetas['theta_0'] - learning_rate * _p0, 
            'theta_1':thetas['theta_1'] - learning_rate * _p1
        }
    return thetas
```
code to learn on multiple variables:
```python
import numpy as np
import pandas as pd

def learn_multiple_y(df:pd.DataFrame, x_col:str, y_col:str, epoch:int, learning_rate:float)->dict:
    model = {}
    params = list(df[y_col].unique())
    
    for param in params:
        _df = df.copy(deep=True)
        _df[y_col] = _df[y_col].replace(params, [1 if e == param else 0 for e in params])
        model[param] = gradient_descent(_df[x_col], _df[y_col], epoch, learning_rate)

    return model
```