from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


class NaiveBayes:
    
    @classmethod
    def run(cls, x_train, y_train):

        cls.__nb = GaussianNB()
        cls.__nb.fit(x_train, y_train)
    
    @classmethod
    def predict(cls, x_test):

        return cls.__nb.predict(x_test)


class DecisionTree:

    @classmethod
    def run(cls, x_train, y_train, criterion='entropy'):

        cls.__dt = DecisionTreeClassifier(criterion=criterion, random_state=0)
        cls.__dt.fit(x_train, y_train)
    
    @classmethod
    def predict(cls, x_test):

        return cls.__dt.predict(x_test)


class RandomForest:
    
    @classmethod
    def run(cls, x_train, y_train, n_tree, criterion='entropy'):

        cls.__rf = RandomForestClassifier(n_estimators=n_tree, criterion=criterion, random_state=0)
        cls.__rf.fit(x_train, y_train)
    
    @classmethod
    def predict(cls, x_test):

        return cls.__rf.predict(x_test)