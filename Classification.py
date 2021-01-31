
import numpy.random as numrandom
from Graphs import view
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn import metrics
import warnings

def classfy():
    # Split the data into columns and read
    warnings.warn("Variables are collinear.")
    datainput = pd.read_csv("trained.csv")
    # Set the outcome and dedlete it
    y = datainput['Level']
    del datainput['Level']
    # Split data into Test & Training set where test data is 30% & raining data is 70%
    x_train, x_test, y_train, y_test = train_test_split(datainput, y, test_size=0.3)

    # Next use Bayesian Classifier
    classify3 = BernoulliNB()
    # Train the model
    classify3.fit(x_train, y_train)
    # Use the model on the test data
    predicted3 = classify3.predict(x_test)
    nb = metrics.accuracy_score(y_test, predicted3) * 100
    print("The accuracy score using the Naive Bayes Classifier is ->")
    print(metrics.accuracy_score(y_test, predicted3))
    print('---------------------------------------------- ')

    # Next use FLDA Classifier
    classify4 = LinearDiscriminantAnalysis()
    # Train the model
    classify4.fit(x_train, y_train)
    # Use the model on the test data
    predicted4 = classify4.predict(x_test)
    ld = metrics.accuracy_score(y_test, predicted4) * 100
    print("The accuracy score using the FLDA is ->")
    print(metrics.accuracy_score(y_test, predicted4))
    print('---------------------------------------------- ')
    # Next use SVM
    classify5 = svm.LinearSVC()
    # Train the model
    classify5.fit(x_train, y_train)
    # Use the model on the test data
    predicted5 = classify5.predict(x_test)
    svmdt = metrics.accuracy_score(y_test, predicted5) * 100
    print("The accuracy score using the svm is ->")
    print(metrics.accuracy_score(y_test, predicted5))
    print('---------------------------------------------- ')
    list = []
    list.clear()
    list.append(nb)
    list.append(ld)
    list.append(svmdt)
    view(list)
classfy()



