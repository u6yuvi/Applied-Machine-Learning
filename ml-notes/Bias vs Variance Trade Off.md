Bias vs Variance Trade Off

1. Definition
   1. Equation relating bias and variance

**Bias** is model error from under-fitting data. Having high bias means that the machine learning model is too simple and does not capture the relationship between the features and the target. Logistic regressions are likely to have higher bias than a neural network.

**Variance** is model error from overfitting data. Having high variance means that a model is overly sensitive to changes in training data and is therefore capturing too much noise. Therefore, the model is too complex for the problem it is trying to address. A neural network is likely to have higher variance than logistic regression.

The decomposition is represented by the equation:

```
Model error = Bias error (squared) + Variance error + Irreducible error
```

Irreducible error is error that cannot be addressed. For instance, one type of irreducible error would be noise in measurements of data. The bias-variance trade-off addresses the balance between the two. Flexible models have low bias and high variance, whereas more rigid models have high bias and low variance. Each model has relevant usage depending on the desired objectives of the model.

For logistic regression, any number of answers involving general practices to tune models would suffice. Some relevant answers include:

1. **Hyperparameter tuning**: using grid search to tune parameters of the logistic regression. Note that it is important to use cross validation here to prevent introducing more variance.
2. **Ensembles**: since logistic regression looks at linear decision boundaries, it may be the case that classes are not linearly separable so using tree-based approaches or SVMs could be useful.
3. **Feature normalization and scaling**: it is important to normalize and scale features to avoid their weights dominating the model.
4. **Adding more features**: since logistic regression is high bias, adding more features should be helpful (in the movie recommendation domain, it could be items outside of user-movie pairs that have been watched, such as their reviews or ratings of movies).

For neural networks, relevant answers include:

1. **Trying different network architectures**: more dense or sparse networks may be more suitable, and there is likely to be a good component of “memory” involved via recurrent neural networks.
2. **Weight regularization**: penalizing coefficients to reduce overfitting of the neural network.
3. **Adding more features**: similar to the above case with logistic regression, by integrating other types of information, for example: what a user has been querying recently, social media data, etc.
4. **Different cost functions or data labels**: it can be useful to make sure that the problem setup is correct to be applying a neural network for. Otherwise there may need to be a different way of encoding the problem.