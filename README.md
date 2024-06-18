# ft_dslr
---
## Usage
### For the required data analysis:

_Move to the corresponding directory (in srcs)_
- Describe:
  - Run `python describe.py <file>`
- Histogram:
  - Run `python histogram.py <file>`
- Scatter plot:
  - Run `python scatter_plot.py <file>`
- Pair plot:
  - Run `python pair_plot.py <file>`

All script came with its associate jupyter notebook witch give us more information and answers for
the questions of the subject.

### For the required training and predict process:

_Move the directory : srcs/logistic_regression_

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

To find the best param $$\beta_0$$ and $$\beta_1$$ we search to minimise the cost function associate
to the sigmoid.

Bernoulli distribution:
```math
    P(y = Y) = \sigma(x)^{y} * (1 - \sigma(x))^{(1 - y)}
```
