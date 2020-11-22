# Machine Learning Algorithms with Scikit-Learn

> Comparion of several machine learning algorithms in the Census dataset.

### Census Dataset

Dataset with ``32561 samples``. 

``Objective``: Predict if the income is ``<=50K`` or ``>50K``.

Outputs:

- ``0``: <=50K.
- ``1``: >50K.

**Loading Dataset**

- `split_data`: Splitting of the dataset in training and testing.
- `label_encoder`: Transforms categorical data into numeric data. The value _True_ is the default.
- `one_hot`: Creates one hot vector in categorical data. The value _False_ is the default.
- `std`: Applies StandardScaler normalization. The value _False_ is the default.

Example, using `one_hot` and `std`:

```python
from dataset import Census
x_train, y_train, x_test, y_test = Census.load_data(split_data=0.15, label_encoder=False, one_hot=True, std=True)
```

### Results

**Naive Bayes**

Training and Testing:

```python
from ml import NaiveBayes

NaiveBayes.run(x_train, y_train)  # Training
prediction = NaiveBayes.predict(x_test)  # Testing
```

Accuracy (`split_data=0.15`):

- label_encoder: 0.7953
- label_encoder + one_hot: 0.7951
- label_encoder + std: 0.8057
- label_encoder + one_hot + std: 0.4768

**Decision Tree**

Training and Testing:

```python
from ml import DecisionTree

DecisionTree.run(x_train, y_train, criterion='entropy')  # Training
prediction = DecisionTree.predict(x_test)  # Testing
```

Accuracy (`split_data=0.15`):

- label_encoder: 0.8129
- label_encoder + one_hot: 0.8102
- label_encoder + std: 0.8129
- label_encoder + one_hot + std: 0.8104

**Random Forest**

Training and Testing:

```python
from ml import RandomForest

RandomForest.run(x_train, y_train, n_tree=40, criterion='entropy')  # Training
prediction = RandomForest.predict(x_test)  # Testing
```

Accuracy (`split_data=0.15`):

- label_encoder: 0.8481
- label_encoder + one_hot: 0.8489
- label_encoder + std: 0.8483
- label_encoder + one_hot + std: 0.8477


**KNN**

Training and Testing:

```python
from ml import KNN

KNN.run(x_train, y_train, k=5)  # Training
prediction = KNN.predict(x_test)  # Testing
```

Accuracy (`split_data=0.15`):

- label_encoder: 0.7746
- label_encoder + one_hot: 0.7760
- label_encoder + std: 0.8219
- label_encoder + one_hot + std: 0.8223


### Contact

[linkedin.com/in/edvaldo-melo/](https://www.linkedin.com/in/edvaldo-melo/)

emeloppgi@gmail.com

[github.com/EFMelo](https://github.com/EFMelo)