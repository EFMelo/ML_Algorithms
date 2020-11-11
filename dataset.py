from pandas import read_csv
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class Census:

    @classmethod
    def load_data(cls, split_data, label_encoder=True, one_hot=False, std=False):
        
        data = read_csv('census.csv')

        cls.__x = data.iloc[:, 0:14].values
        cls.__y = data.iloc[:, 14].values

        # Pre-processing
        if one_hot:
            label_encoder = False
            cls.__one_hot_encoder()

        if label_encoder:
            cls.__lbl_encoder()

        if std:
            cls.__x = StandardScaler().fit_transform(cls.__x)  # normalization


        # Splitting data into training and testing
        x_train, x_test, y_train, y_test = train_test_split(cls.__x, cls.__y, test_size=split_data, random_state=0)

        return x_train, y_train, x_test, y_test

        
    @classmethod
    def __lbl_encoder(cls):

        # input data
        cls.__x[:, 1] = LabelEncoder().fit_transform(cls.__x[:,1])
        cls.__x[:, 3] = LabelEncoder().fit_transform(cls.__x[:,3])
        cls.__x[:, 5] = LabelEncoder().fit_transform(cls.__x[:,5])
        cls.__x[:, 6] = LabelEncoder().fit_transform(cls.__x[:,6])
        cls.__x[:, 7] = LabelEncoder().fit_transform(cls.__x[:,7])
        cls.__x[:, 8] = LabelEncoder().fit_transform(cls.__x[:,8])
        cls.__x[:, 9] = LabelEncoder().fit_transform(cls.__x[:,9])
        cls.__x[:, 13] = LabelEncoder().fit_transform(cls.__x[:,13])

        # output data
        cls.__y = LabelEncoder().fit_transform(cls.__y)

    
    @classmethod
    def __one_hot_encoder(cls):

        ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [1, 3, 5, 6, 7, 8, 9, 13])], remainder='passthrough')
        cls.__x = ct.fit_transform(cls.__x).toarray()        