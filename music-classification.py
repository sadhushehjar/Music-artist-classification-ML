# Started by: Shehjar Sadhu.
# Music classification using machine learning techniques.
# Modles used: KNN,....?

# How to run: python3 music-classification.py millionSongs.csv

####TO DO'S:#####
#1. Proper training for KNN -- try out different neighbours.
#2. Write function for SVM,logistic regression  and doing proper training.
#3. May be we should take in classifier names as commandline arguments.
#4. Make the hyperparameter json file for the KNN hyper parameters.

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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

# Returns a list of predictions.
def classifierKNN(X_train, y_train,X_test,float_dataset, y,df_json):

    #Classifier from sklearn.
    classifier = KNeighborsClassifier(n_neighbors=5,weights='uniform')#.fit(X_train, y_train)

    #Does 10-fold cross-validation.
    #scores = cross_val_score(classifier, float_dataset, y, cv=10)
    #print(scores)

    #Best accuracy achieved: 13.636363636363635 %
    #print("Best accuracy a8chieved:",scores.max()*100,"%")
    #Grid Search
    print("# Tuning hyper-parameters for")
    print(param_grid['n_neighbors'])
    print()

    clf = GridSearchCV(classifier, df_json, cv=10,
                       scoring='accuracy')
    clf.fit(X_train, y_train)
    print("Best parameters set found on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    print("Cross Validation Results: ",means,"\n")
    print("Best Score: ",clf.best_score_)
    print(clf.best_params_)
    print()
    plt.plot(param_grid['n_neighbors'], means)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
    #y_test, y_pred = y_test, clf.predict(X_test)

def support_vector_machine(X_train, y_train,X_test,float_dataset):

    parameters = [{'kernel': ['rbf'],
                   'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                   'C': [1, 10, 100, 1000]},
                   {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', verbose=1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("predictions \n",predictions)
    print("#Tuning hyper-parameters for")
    print(parameters)
    print()

    clf = GridSearchCV(clf, parameters, cv=2,
                       scoring='accuracy',n_jobs=-1)
    clf.fit(X_train, y_train)


    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("mean cross-validation accuracy: ",mean,"for",params)
        print()


    print("Best Score: ",clf.best_score_)
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def naive_bayes(X_train,Y_train,X_Test,Y_Test):

    clf = GaussianNB()
    clf.fit(X_train,Y_train)
    predictions = clf.predict(X_Test)
    testPredictions = clf.predict(X_train)
    print("Gaussian Prediction Accuracy on test data: ",accuracy_score(predictions,Y_Test))
    print("Gaussian Prediction Accuracy on train data: ",accuracy_score(testPredictions,Y_train))

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

# return y_pred
'''def svmClassifier(X_train, y_train,X_test,float_dataset):

    #svm Classifier.
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train) #svm.SVC(gamma='scale', kernel='rbf', decision_function_shape='ovo').fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(y_pred)
    #Does 10-fold cross-validation.
    scores = cross_val_score(classifier, float_dataset, y, cv=10)
    print(scores)

    #Best accuracy achieved: ----- %
    print("Best accuracy achieved:",scores.max()*100,"%")
    return y_pred'''


if __name__ == '__main__':

    #Takes in commandline argument for the name of the data set.
    f_name = sys.argv[1]
    json_file = sys.argv[2]
    #....for json config files.....#
    #with open(json_file, 'r') as f:
        #df_json = json.load(f)
    #Contains only 10,000 instances. Original -> 1 Million!
    dataset = pd.read_csv(f_name) #Shape: 10000x35.

    y = dataset[dataset.columns[2]].values
    X_train, y_train, X_test,y_test,float_dataset  = preprocessing(dataset)
    #print("X_train",X_train, "y_train",y_train, "X_test",X_test,"y_test",y_test)
    ########-----------------------------KNN------------------------------------#######
    #Hyperparameters::
    #print("Hyperparameters: ",df_json)
    param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
    117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160,
    161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178,
    179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196,
    197, 198, 199]}

    #start_time = time.time()
     # Classification:
    #classifierKNN(X_train, y_train,X_test,float_dataset, y,param_grid)
    #print("--- %s seconds ---" % (time.time() - start_time))

    #for i in y_preds: #2000 predictions.
    #    print(i)


    ########-----------------------------SVM-------------------------------------#######
    start_time = time.time()
    support_vector_machine(X_train, y_train,X_test,float_dataset)
    print((time.time() - start_time)*0.0166667,"Min")



    #y_preds_svm = svmClassifier(X_train, y_train,X_test,float_dataset,param_grid)
    #print("########--------------------------SVM--------------------------########")
    #for k in y_preds_svm: #2000 predictions.
    #    print(k)
