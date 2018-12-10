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
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
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

    # X_train: 8000x23, y_train: 8000x1 X_test: 2000x23.
    return X_train, y_train, X_test, y_test,float_dataset

# Returns a list of predictions.
def classifierKNN(X_train, y_train,X_test,float_dataset, y,n_neigh,algo,p_dist,dist_metric,weights):

    #Classifier from sklearn.
    classifier = KNeighborsClassifier( n_neighbors=13,algorithm = "kd_tree"
                                      ,p = 2,metric="minkowski"
                                      ,weights="distance").fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    #Does 10-fold cross-validation.
    scores = cross_val_score(classifier, float_dataset, y, cv=10)
    print(scores)

    #Best accuracy achieved: 13.636363636363635 %
    print("Best accuracy achieved:",scores.max()*100,"%")
    return y_pred


if __name__ == '__main__':

    #Takes in commandline argument for the name of the data set.
    f_name = sys.argv[1]

    #Contains only 10,000 instances. Original -> 1 Million!
    dataset = pd.read_csv(f_name) #Shape: 10000x35.

    y = dataset[dataset.columns[2]].values
    X_train, y_train, X_test,y_test,float_dataset  = preprocessing(dataset)

    ########-----------------------------KNN------------------------------------#######

    #Hyperparameters::
    n_neigh=13
    algo="kd_tree"
    p_dist = 2
    dist_metric="minkowski"
    weights="distance"

    # Classification:
    y_preds = classifierKNN(X_train, y_train,X_test,float_dataset, y,n_neigh,algo,p_dist,dist_metric,weights)

    for i in y_preds: #2000 predictions.
        print(i)
