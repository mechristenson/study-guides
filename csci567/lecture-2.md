# CSCI 567: Machine Learning
## Midterm Study Guide
## Lecture 2

### Major Topics:
  - Machine Learning Overview
  - Nearest Neighbor Classifier
  - Core Machine Learning Concepts
  - Steps of Developing a Machine Learning System
  - Linear Regression
  - Regression with Non-Linear Basis
  - Regularized Regression
  - Linear Discriminant Analysis
  - Perceptron
  - Logistic Regression
  - Multi-Class Classification
  - Neural Networks
  - Convolutional Neural Networks
  - Kernel Methods
  - Lagrangian Duality
  - Support Vector Machines

### Sources of Information
  - CSCI 567 Lecture Notes, Haipeng Luo [Lecture]
  - CSCI 567 Discussion Notes, Victor Adamchik [Discussion]
  - Machine Learning a Probabilistic Perspective, Kevin Murphy [MLaPP]
  - The Elements of Statistical Learning, Hastie, Tibshirani, Friedman [ESL]

---

### Lecture 2 Notes

#### Topics
  - Linear Regression
  - Regression with Non-Linear Basis
  - Regularized Regression

#### Sources
  - Lecture 2 (Week 2)
  - Discussion 2 (Week 3)
  - MLaPP 1.4.6, 4.2.1-5. 8.1-3, 8.5.1-4
  - ESL 4.1-2, 4.4

#### Definitions
  1. Regression
    - Predicting a continuous outcome variable using past observations

#### Concepts

##### I. Regression vs. Classification
  - Regression predicts a continuous outcome
  - Classification predicts a discrete outcome
  - Prediction errors are measured differently
  - Learning algorithms are different

#### Algorithms

##### I. Linear Regression
  Predicting a continuous outcome variable using past observations with linear models.

  Basic linear models include simple y = mx + b formulas, and it is up to us to determine what (and how many) parameters to use.

###### How to learn unknown parameters
  Pick a model (unknown parameters) that minimizes average/total prediction error on the training set. Note that we would ideally like to minimize the error on the test set, but we cannot use the test set while training.

  Note that classification error, (0-1 loss) does not work in this case, where we have a continuous outcome. The probability of exactly guessing the outcome is 0 so we must use a different type of error. We can use:
  - absolute error: |prediction - actual|
  - squared error: (prediction - actual)<sup>2</sup>, squared error is most common

###### Formal Setup for Linear Regression
  - Input: x ∈ ℝ<sup>D</sup> (features, covariates, predictors, etc.)
  - Output: y ∈ ℝ (responses, targets, outcomes, etc.)
  - Training Data: D = {(x<sub>n</sub>, y<sub>n</sub>, n = 1, 2,..., N)}
  - Linear Model: f: ℝ<sup>D</sup>→ℝ, with f(x) = w<sub>0</sub> + Σ<sup>D</sup><sub>d=1</sub> w<sub>d</sub>x<sub>d</sub> = w<sub>0</sub> + ** w<sup>T</sup>** x
    - i.e. this is a hyper-plane parametrized by:
      - ** w ** = [w<sub>1</sub> w<sub>2</sub> ... w<sub>D</sub>]<sup>T</sup>, the weight vector
      - bias w<sub>0</sub>
      - Essentially, w<sub>0</sub> is the intercept and  ** w<sup>T</sup>** is the slope
    - Note: for convenience, we:
      - append 1 to each x as the first feature: x = [1 x<sub>1</sub> x<sub>2</sub> ... x<sub>D</sub>]<sup>T</sup>
      - let ** w ** = [w<sub>0</sub> w<sub>1</sub> w<sub>2</sub> ... w<sub>D</sub>]<sup>T</sup>
      - therefore our model simplifies to: f(x) = ** w<sup>T</sup>** x

###### Goal of Linear Regression
Minimize total squared error
  - Residual Sum of Squares (RSS), a function of w:
    - RSS(w) = Σ<sub>n</sub>(f(x<sub>n</sub>) - y<sub>n</sub>)<sup>2</sup> = Σ<sub>n</sub>(x<sub>n</sub><sup>T</sup>w - y<sub>n</sub>)<sup>2</sup>
  - Find w* = argmin<sub>w∈ℝ<sup>D+1</sup></sub> RSS(w), i.e least (mean) squares solution (more generally called empirical risk minimizer)

In essence, we reduce machine learning to an optimization problem. In principle, we can apply any optimization algorithm, but linear regression admits a closed form solution were we can determine the exact value of w* based on our training data.

###### Learning Parameters
D = 0: If we have only one parameter to learn, w<sub>0</sub>, so f(x) = w<sub>0</sub>, we should set w<sub>0</sub> equal to the mean of our data. In this case, our optimization objective becomes:
  - RSS(w<sub>0</sub>) = Σ<sub>n</sub>(w<sub>0</sub> - y<sub>n</sub>)<sup>2</sup>, (the quadratic aw<sub>0</sub><sup>2</sup> + bw<sub>0</sub> + c)
    - = N w<sub>0</sub><sup>2</sup> - 2 (Σ<sub>n</sub> y<sub>n</sub>) w<sub>0</sub> + cnt, since cnt does not depend on w<sub>0</sub>, the optimization objective, we can just disregard it as a constant
    - = N(w<sub>0</sub> - 1/N Σ<sub>n</sub> y<sub>n</sub>)<sup>2</sup> + cnt
  - Therefore w*<sub>0</sub> = 1/N Σ<sub>n</sub> y<sub>n</sub>, i.e. the average
  - Note, if we use absolute error instead of squared error, the minimizer is the median

D = 1: If we have two parameters to learn, then our optimization objective becomes:
  - RSS(w) = Σ<sub>n</sub>(w<sub>0</sub> + w<sub>1</sub>x<sub>n</sub> - y<sub>n</sub>)<sup>2</sup>
  - Now, we want to find stationary points, or points with zero gradient:
    - ∂RSS(w)/∂w<sub>0</sub> = 0 ⇒ 2Σ<sub>n</sub>(w<sub>0</sub> + w<sub>1</sub>x<sub>n</sub> - y<sub>n</sub>) = 0 ⇒ Σ<sub>n</sub>(w<sub>0</sub> + w<sub>1</sub>x<sub>n</sub> - y<sub>n</sub>) = 0, and
    - ∂RSS(w)/∂w<sub>1</sub> = 0 ⇒ 2Σ<sub>n</sub>(w<sub>0</sub> + w<sub>1</sub>x<sub>n</sub> - y<sub>n</sub>)x<sub>n</sub> = 0 ⇒ Σ<sub>n</sub>(w<sub>0</sub> + w<sub>1</sub>x<sub>n</sub> - y<sub>n</sub>)x<sub>n</sub> = 0
  - These equations become a linear system:
    - Nw<sub>0</sub> + w<sub>1</sub>Σ<sub>n</sub>x<sub>n</sub> = Σ<sub>n</sub>y<sub>n</sub>, and
    - w<sub>0</sub>Σ<sub>n</sub>x<sub>n</sub> + w<sub>1</sub>Σ<sub>n</sub>x<sup>2</sup><sub>n</sub> = Σ<sub>n</sub>y<sub>n</sub>x<sub>n</sub>
  - We can put these in 2d matrix form (excuse ugly format):
    - (N &nbsp; &nbsp; &nbsp; Σ<sub>n</sub>x<sub>n</sub> &nbsp;)  (w<sub>0</sub>) = (Σ<sub>n</sub>y<sub>n</sub> &nbsp; &nbsp;)
    - (Σ<sub>n</sub>x<sub>n</sub> &nbsp; Σ<sub>n</sub>x<sup>2</sup><sub>n</sub>) (w<sub>0</sub>) = (Σ<sub>n</sub>y<sub>n</sub>x<sub>n</sub>)

**** STOPPED AT PAGE 23 ****


---

### Math Review

#### Concepts
