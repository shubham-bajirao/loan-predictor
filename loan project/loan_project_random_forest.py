""

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score


#Importing datasets
dataset = pd.read_csv('loan_project_dataset.csv')
X = dataset.iloc[:, 3:12].values
y = dataset.iloc[:, 12].values

#Take care of missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 5:6])
X[:, 5:6] = imputer.transform(X[:, 5:6])

imputer1 = SimpleImputer(missing_values= np.nan, strategy = 'most_frequent')
imputer1 = imputer1.fit(X[:, 0:5])
X[:, 0:5] = imputer1.transform(X[:, 0:5])
imputer1 = imputer1.fit(X[:, 6:9])
X[:, 6:9] = imputer1.transform(X[:, 6:9])

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
'''labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
'''


ct = ColumnTransformer([("Country", OneHotEncoder(), [0,1,2])], remainder = 'passthrough')
X = ct.fit_transform(X)
X = X.astype(int)
X = X[:, 1:]


labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
