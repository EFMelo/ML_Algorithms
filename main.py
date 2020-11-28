from dataset import Census
from ml import SVM
from sklearn.metrics import accuracy_score

# loading data
x_train, y_train, x_test, y_test = Census.load_data(split_data=0.15, std=True)

SVM.run(x_train, y_train, c=1, kernel='linear')  # Training
prediction = SVM.predict(x_test)  # Testing

acc = accuracy_score(y_test, prediction)  # accuracy
print(acc)