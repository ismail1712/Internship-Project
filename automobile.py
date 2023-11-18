# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
dataset = pd.read_csv('auto\CarPrice_project.csv')

dataset = dataset.replace({'?': np.NaN, 'n.a': np.NaN})


dataset.drop(columns=['fuelsystem'],axis=1,inplace=True)

def doors(doornumber):
  if doornumber == 'four':
    numdoors = 4
    return numdoors
  else:
    numdoors = 2
    return numdoors
dataset['numdoors'] = dataset.apply(lambda x: doors(x['doornumber']),axis=1)

dataset.drop(columns=['doornumber'],axis=1,inplace=True)


dataset.drop(columns=['symboling'],axis=1,inplace=True)

dataset.drop(columns=['car_ID'],axis=1,inplace=True)

dataset.drop(columns=['CarName'],axis=1,inplace=True)

dependent_variable = 'price'
columns=['enginesize','wheelbase','peakrpm','horsepower','citympg','numdoors']
X=dataset[columns]
y=dataset[dependent_variable]

X_train, X_test, y_train, y_test = train_test_split( X,y , test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)


pickle.dump(regressor, open('auto\model.pkl', 'wb'))