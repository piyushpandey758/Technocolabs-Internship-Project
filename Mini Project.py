import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = pd.read_csv('parkinsons.data')
#dataset_updrs = pd.read_csv('parkinsons_updrs.data')

#Get the features and labels
features=dataset.loc[:,dataset.columns!='status'].values[:,1:]
labels=dataset.loc[:,'status'].values

# Scale continuous data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
preprocess = StandardScaler()

features = preprocess.fit_transform(features)

# Split in train/test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)

print(X_test)
print(X_train)
print(y_test)
print(y_train)