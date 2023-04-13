import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from keras.models import Sequential
from keras.layers import Dense
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.model_selection import KFold
import tensorflow as tf

data_path = os.path.join(os.getcwd(), '../data/heart.csv')
df = pd.read_csv(data_path)

X = df.drop('output', axis=1)
y = df['output']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0) 

numeric = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
ordinal = ['caa']
categorical = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'thall']

numeric_transformer = Pipeline([('inputer', SimpleImputer(strategy='mean')),
                               ('scaler', StandardScaler())
                               ])
cat_transformer = Pipeline([('inputer', SimpleImputer(strategy='most_frequent')),
                             ('onehot', OneHotEncoder())
                            ])
ordinal_transformer = Pipeline([('inputer', SimpleImputer(strategy='most_frequent')),
                               ('ord_encod', OrdinalEncoder())
                               ])

preprocessor = ColumnTransformer([('num', numeric_transformer, numeric),
                                ('ordinal', ordinal_transformer, ordinal),
                                ('cat', cat_transformer, categorical),
                                ])

scaler = preprocessor.fit(X_train)
X_train_trans = scaler.transform(X_train)
X_test_trans = scaler.transform(X_test)

df_transf = pd.DataFrame(X_train_trans)

np.random.seed(0)
tf.random.set_seed(0)

def create_model_grid(activation='relu', optimizer='rmsprop', neurons=20):
    model = Sequential()
    model.add(Dense(neurons, input_dim=26, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model_grid = KerasClassifier(model=create_model_grid, activation='relu', neurons=10, #add parameters from model not included in KerasClassifier
                        epochs=10, 
                        batch_size=32, 
                        verbose=0, random_state=0)

activations = ['tanh','relu','sigmoid']
neurons = [10,15,20,26,30,35]

param_grid = dict(activation=activations, 
                  neurons=neurons)

from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(estimator=model_grid, 
                    param_grid=param_grid,
                    cv =4,
                    verbose=1, scoring='accuracy')

grid = grid.fit(X_train_trans, y_train)


model = grid.best_estimator_
model_training_score = model.score(X_train_trans, y_train)
model_test_score = model.score(X_test_trans, y_test)
print(model_test_score)

def create_model_grid(activation='relu', neurons_l1=20, neurons_l2=20):
    model = Sequential()
    model.add(Dense(neurons_l1, input_dim=26, activation=activation))
    model.add(Dense(neurons_l2, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_grid_1 = KerasClassifier(model=create_model_grid, activation='relu', neurons_l1=20, neurons_l2=20,
                        batch_size=32, 
                        verbose=0, random_state=0)

activations = ['tanh','relu','sigmoid']
neurons_l1 = [10,15,20,26,30]
neurons_l2 = [10,15,20,26,30]

param_grid_1 = dict(activation=activations, 
                  neurons_l1=neurons_l1,
                 neurons_l2=neurons_l2)

grid_1 = GridSearchCV(estimator=model_grid_1, 
                    param_grid=param_grid_1,
                    cv = 4,
                    verbose=1, scoring='accuracy')

grid_1 = grid_1.fit(X_train_trans, y_train)

model_1 = grid_1.best_estimator_
model_1_training_score = model_1.score(X_train_trans, y_train)
print(model_1_training_score)
model_1_test_score = model_1.score(X_test_trans, y_test)
print(model_1_test_score)

def create_model_grid(activation='relu', neurons_l1=20, neurons_l2=20, neurons_l3=20):
    model = Sequential()
    model.add(Dense(neurons_l1, input_dim=26, activation=activation))
    model.add(Dense(neurons_l2, activation=activation))
    model.add(Dense(neurons_l3, activation=activation))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', metrics=['accuracy'])
    return model

model_grid_2 = KerasClassifier(model=create_model_grid, activation='relu', neurons_l1=20, neurons_l2=20, neurons_l3=20,
                        batch_size=32, 
                        verbose=0, random_state=0)

activations = ['tanh','relu','sigmoid']
neurons_l1 = [10,15,20,26,30]
neurons_l2 = [10,15,20,26,30]
neurons_l3 = [10,15,20,26,30]

param_grid_2 = dict(activation=activations, 
                  neurons_l1=neurons_l1,
                  neurons_l2=neurons_l2,
                  neurons_l3=neurons_l3)

grid_2 = GridSearchCV(estimator=model_grid_2, 
                    param_grid=param_grid_2,
                    cv = 4,
                    verbose=1, scoring='accuracy')

grid_2 = grid_2.fit(X_train_trans, y_train)

model_2 = grid_2.best_estimator_
model_2_training_score = model_2.score(X_train_trans, y_train)
model_2_test_score = model_2.score(X_test_trans, y_test)
print(model_2_test_score)

models = ['una capa oculta', 'dos capas ocultas', 'tres capas ocultas']
test_scores = [model_test_score, model_1_test_score, model_2_test_score]
training_scores = [model_training_score, model_1_training_score, model_2_training_score]

