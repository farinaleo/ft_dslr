# ft_dslr

## 🚀 Principal Functionalities

### 📊 Dataset Description Script

Dive deep into your data with the **Dataset Description Script**! 🌟

- **Command-Line Interface**: Easily specify your dataset file and toggle options like dropping empty columns, adding bonus info, and enabling verbose mode.
- **Statistical Summary**: Get a comprehensive overview with stats like count, mean, standard deviation, min, max, quantiles, missing values, and range for numerical columns.
- **Flexibility**: Customize your output with bonus information and detailed process insights.

### 📈 Histogram Plotting Script

Visualize your data distributions effortlessly with the **Histogram Plotting Script**! 🎨

- **Command-Line Interface**: Select your dataset file, choose subjects to plot, enable verbose mode, and use all subjects as input.
- **Customizable Plots**: Generate color-coded histograms by Hogwarts Houses for a clear visual summary.
- **Visualization**: Understand data distribution across different categories with ease.

### 🔍 Pair Plot Script

Explore relationships in your dataset with the **Pair Plot Script**! 🌐

- **Command-Line Interface**: Specify your dataset file and enable verbose mode for detailed insights.
- **Comprehensive Visualization**: Create a grid of scatter plots and histograms, color-coded by Hogwarts Houses, to analyze relationships and distributions.
- **Detailed Analysis**: Gain a deeper understanding of your data's intricate connections.

### 📉 Scatter Plot Script

Uncover correlations with the **Scatter Plot Script**! 🔍

- **Command-Line Interface**: Specify your dataset file, enable verbose mode, and plot all numerical column pairs with the `--all` flag.
- **Flexible Plotting**: Visualize relationships between variables, differentiated by Hogwarts Houses.
- **Visual Analysis**: Identify trends and patterns effortlessly.

## 🎯 Logistic Regression Training Script

Train your logistic regression model with ease using the **Logistic Regression Training Script**! 🚀

- **Command-Line Interface**: Easily configure your training process with options for dataset file, verbosity, accuracy computation, learning rate, epochs, validation ratio, seed, save directory, and batch selection.
- **Model Training**: Train a logistic regression model to predict Hogwarts Houses using various batch selection methods like mandatory, stochastic, mini, and random step.
- **Performance Metrics**: Evaluate your model's accuracy and save the trained model for future predictions.
- **Flexibility**: Customize your training process with configurable parameters and save your model for later use.

## 🔮 Logistic Regression Prediction Script

Predict Hogwarts Houses effortlessly with the **Logistic Regression Prediction Script**! 🔮

- **Command-Line Interface**: Specify your dataset file, model weights, configuration file, and destination path for predictions.
- **Model Prediction**: Apply a pre-computed logistic regression model to predict Hogwarts Houses from your dataset.
- **Data Preparation**: Format and prepare your data for prediction, ensuring compatibility with the trained model.
- **Output**: Save the predicted labels to a CSV file, making it easy to analyze and use the results.

---

## 🚀 Usage

### For Data Analysis Adventures 📊

_Navigate to the corresponding directory (in `srcs`) and let the magic happen!_

- **Describe**:
  - Dive into your data with `python ft_dslr/describe/describe.py <file>`

- **Histogram**:
  - Visualize distributions effortlessly with `python ft_dslr/histogram/histogram.py <file>`

- **Scatter Plot**:
  - Uncover relationships with `python ft_dslr/scatter_plot/scatter_plot.py <file>`

- **Pair Plot**:
  - Explore multivariate relationships using `python ft_dslr/pair_plot/pair_plot.py <file>`

Each script comes with an associated Jupyter notebook, providing deeper insights and answers to your data questions!

### For Training and Prediction Quests 🎯

_Harness the power of logistic regression with these essential scripts:_

- **Training**:
  - Train your model to predict Hogwarts Houses with `logreg_train.py`.
  - Ensure your model's accuracy is 98% or higher for top-notch predictions!

- **Prediction**:
  - Predict student houses using your trained model with `logreg_predict.py`.
  - Aim for a maximum error rate of just 2% during predictions.

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
    {\partial \mathcal{L} \over \partial \beta_1} = {\partial \mathcal{L} \over \partial \sigma}
     * {\partial \sigma \over \partial \beta_1}
     = 1 / m * \sum^{n - 1}_{i = 0} (\sigma(x^{(i)}) - y^{(i)}) * x^{(i)}
```
and
```math
    {\partial \mathcal{L} \over \partial \beta_0} = {\partial \mathcal{L} \over \partial \sigma}
     * {\partial \sigma \over \partial \beta_0}
     = 1 / m * \sum^{n - 1}_{i = 0} (\sigma(x^{(i)}) - y^{(i)})
```
