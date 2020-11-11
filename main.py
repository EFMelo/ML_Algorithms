from dataset import Census
from ml import RandomForest
from sklearn.metrics import accuracy_score

# loading data
x_train, y_train, x_test, y_test = Census.load_data(split_data=0.15)

RandomForest.run(x_train, y_train, n_tree=40)  # Training
prediction = RandomForest.predict(x_test)  # Testing

acc = accuracy_score(y_test, prediction)  # accuracy
print(acc)