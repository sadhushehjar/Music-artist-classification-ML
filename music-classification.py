# Started by: Shehjar Sadhu and Mikel Gjergji.

# Music classification using machine learning techniques.
# Modles used: K-nearest neighbours, support vector machines, Logistic regression, Naive bayes

# How to run: python3 new-music-classification.py test-subset-millionsongs.csv

#imports
import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Returns formatted X_train, y_train, X_test so they sont have any null values or strings in X.
def preprocessing(dataset):

    #Contains only floats, (Release id and year - ints) but not included.
    float_dataset = dataset.select_dtypes(include=['float']) #Shape: 10000x23

    #Fill the missing values in each column with the mean of the corresponding row.
    for c in float_dataset.columns:
        float_dataset[c].fillna(float_dataset[c].mean(), inplace=True)
    print(float_dataset.head())

    #Double check if there are any missing valuesself.
    #False ==> No missing values.
    print("Any missing values for X: ",float_dataset.isnull().values.any())

    #Artist names.
    y = dataset[dataset.columns[2]].values #10000 names.

    #Split the dataset into traing and testing.
    X_train , X_test, y_train, y_test = train_test_split(float_dataset, y, test_size = 0.2, random_state = 1)

    #Double checks if there are any missing valuesself in X_train.
    print("Any missing values for X_train: ",X_train.isnull().values.any())

    #print("X_train",X_train, "y_train",y_train, "X_test",X_test,"y_test",y_test)
    # X_train: 8000x23, y_train: 8000x1 X_test: 2000x23.
    return X_train, y_train, X_test, y_test,float_dataset

# Runs sklearns k nearest neighbours and retruns a list of pridections.
# Does tuning hyperparameters such as K and performing grid search on that.
def classifierKNN(X_train, y_train, X_test,y_test,float_dataset,df_json):

    print("Tuning parameters for: ")
    print(df_json['n_neighbors'])
    clf = GridSearchCV(KNeighborsClassifier(), df_json, cv=2,
                       scoring='accuracy')

    clf.fit(X_train, y_train)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("mean cross-validation accuracy: ", mean, "for", params)
        print()

    print("Best Score: ", clf.best_score_)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    predictions = clf.predict(X_test)
    return predictions

#Uses sklearns support vector machine and retruns a list pf predictions.
def support_vector_machine(X_train, y_train,X_test,float_dataset):

    parameters = [{'kernel': ['rbf'],
                   'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                   'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', verbose=1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    return predictions

def naive_bayes(X_train,Y_train,X_Test,Y_Test):

    clf = GaussianNB()
    clf.fit(X_train,Y_train)
    predictions = clf.predict(X_Test)
    testPredictions = clf.predict(X_train)
    print("Gaussian Prediction Accuracy on test data: ",accuracy_score(predictions,Y_Test))
    print("Gaussian Prediction Accuracy on train data: ",accuracy_score(testPredictions,Y_train))
    return testPredictions

def logistic_regression(X_train,y_train):

    parameters = {'C': [.001, .01, .1, 1, 10, 100]}

    clf = GridSearchCV(LogisticRegression(), parameters, cv=2, n_jobs=-1)
    clf.fit(X_train, y_train)

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("mean cross-validation accuracy: ", mean, "for", params)
        print()

    print("Best Score: ", clf.best_score_)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

if __name__ == '__main__':

    #Takes in commandline argument for the name of the data set.
    f_name = sys.argv[1]

    #Contains only 10,000 instances. Original -> 1 Million!
    dataset = pd.read_csv(f_name) #Shape: 10000x35.

    y = dataset[dataset.columns[2]].values
    X_train, y_train, X_test,y_test,float_dataset  = preprocessing(dataset)

    ########-----------------------------KNN------------------------------------------#######

    #Hyperparameters::
    param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    }
    start_time = time.time()
    classifierKNN(X_train, y_train, X_test,y_test,float_dataset,param_grid)
    print("K-nearest neighbours: ",(time.time() - start_time)*0.0166667,"Min")


    ########-----------------------------SVM-------------------------------------------#######
    start_time = time.time()
    support_vector_machine(X_train, y_train,X_test,float_dataset)
    print("support_vector_machine: ",(time.time() - start_time)*0.0166667,"Min")

    ########-----------------------------Naive Bayes-------------------------------------#######
    start_time = time.time()
    naive_bayes(X_train, y_train, X_test,y_test)
    print("Naive bayes: ",(time.time() - start_time)*0.0166667,"Min")

    ########-----------------------------Logistic Regression----------------------------#######
    start_time = time.time()
    logistic_regression(X_train, y_train)
    print("Naive bayes:",(time.time() - start_time)*0.0166667,"Min")
