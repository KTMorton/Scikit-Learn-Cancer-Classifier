from sklearn import neighbors, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from IPython.display import display, HTML
import numpy as np


data_object = pd.read_csv('./breast_cancer_data.csv')

feature_cols = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean','compactness_mean',
                'concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean','radius_se,texture_se',
                'perimeter_se','area_se','smoothness_se','compactness_se','concavity_se','concave points_se','symmetry_se',
                'fractal_dimension_se','radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
                'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst']




trainX = data_object.loc[284:, feature_cols]
trainX = trainX.fillna(0)
trainY = data_object.diagnosis[284:]


testX = data_object.loc[0:284, feature_cols]
testX = testX.fillna(0)
testY = data_object.diagnosis[0:285]






classifier1 =  neighbors.KNeighborsClassifier(weights='distance')
classifier1.fit(trainX, trainY)

prediction_clf1 = classifier1.predict(testX)

print(prediction_clf1)
print(metrics.accuracy_score(testY, prediction_clf1))


classifier2 = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 6), random_state=1,
                            learning_rate='invscaling', max_iter=200)
classifier2.fit(trainX, trainY)

prediction_clf2 = classifier2.predict(testX)

print(prediction_clf2)
print(metrics.accuracy_score(testY, prediction_clf2))





#print(trainX)


