from dataset import Census
from ml import KNN
from sklearn.metrics import accuracy_score

# loading data
x_train, y_train, x_test, y_test = Census.load_data(split_data=0.15, one_hot=True, std=True)

KNN.run(x_train, y_train, k=5)  # Training
prediction = KNN.predict(x_test)  # Testing

acc = accuracy_score(y_test, prediction)  # accuracy
print(acc)