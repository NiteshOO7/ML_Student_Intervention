# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:17:28 2018

@author: nitesh.yadav
"""
import pandas as pd
from time import time
from sklearn.metrics import accuracy_score, make_scorer, f1_score
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

def Data_Load():
    """ loads data from CSV file """
    try:
        full_data_frame = pd.read_csv(r"C:\Users\nitesh.yadav\Desktop\student_intervention\student_data.csv")
        features = full_data_frame.drop('passed', axis = 1)
        labels = full_data_frame['passed']
        print("Student dataset has {} data points with {} variables each.".format(*full_data_frame.shape))
    except FileNotFoundError:
        print("File 'student_data.csv' does not exist, please check the provided path.")
    return full_data_frame, features, labels

def Data_Exploration(full_data_frame):
    """ Explores data for important insights """
    n_students = full_data_frame.shape[0]
    n_features = full_data_frame.shape[1]
    n_passed = full_data_frame.loc[full_data_frame['passed'] == 'yes'].shape[0]
    n_failed = full_data_frame.loc[full_data_frame['passed'] == 'no'].shape[0]
    grad_rate = n_passed / n_students * 100
    print("Total number of students: {}".format(n_students))
    print("Number of features: {}".format(n_features))
    print("Number of students who passed: {}".format(n_passed))
    print("Number of students who failed: {}".format(n_failed))
    print("Graduation rate of the class: {:.2f}%".format(grad_rate))

def Data_Preparation(full_data_frame):
    """ Prepares data for training and testing"""
    feature_columns = list(full_data_frame.columns[: - 1])
    label_column = full_data_frame.columns[- 1]
    print("Feature columns: {}".format(feature_columns))
    print("Target column: {}".format(label_column))
    features = full_data_frame[feature_columns]
    labels = full_data_frame[label_column]
    return features, labels

def Preprocess_Features(features):
    """ Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. """
    preprocessed_features = pd.DataFrame(index = features.index)
    for column, column_data in features.iteritems():
        if column_data.dtype == object:
            column_data = column_data.replace(['yes', 'no'], [1, 0])
        if column_data.dtype == object:
            column_data = pd.get_dummies(column_data, prefix = column)
        preprocessed_features = preprocessed_features.join(column_data)
    print("Processed feature columns ({} total features): {}".format(len(preprocessed_features.columns), list(preprocessed_features.columns)))
    return preprocessed_features

def Train_Classifier(clf, features_train, label_train):
    """Fits a classifier to the training data."""
    start = time()
    clf.fit(features_train, label_train)
    end = time()
    print("Trained model in {:.4f} seconds".format(end - start))
    
def Predict_Labels(clf, features, labels):
    """Makes predictions using a fit classifier based on F1 score."""
    start = time()
    labels_predict = clf.predict(features)
    end = time()
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return accuracy_score(labels, labels_predict)

def Train_Predict(clf, features_train, features_test, labels_train, labels_test):
    """Train and predict using a classifer based on F1 score."""
    print("Training a {} using a training set size of {}:".format(clf.__class__.__name__, len(features_train)))
    Train_Classifier(clf, features_train, labels_train)
    print("Accuracy for training set: {:.4f}.".format(Predict_Labels(clf, features_train, labels_train)))
    print("Accuracy for test set: {:.4f}.".format(Predict_Labels(clf, features_test, labels_test)))
    
def F1_Score(labels, labels_predict):
    """Calculates F1 score"""
    return f1_score(labels, labels_predict, pos_label = 'yes')
    
def Model_Tuning(clf, fatures_train, labels_train):
    """Fine tune the chosen model"""
    cv = StratifiedShuffleSplit(labels_train, random_state = 0)
    scoring_fun = make_scorer(F1_Score)
    params = [{'C': [1, 10, 50, 100, 300, 600, 1000], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5], 'tol':[1e-2, 1e-3, 1e-4, 1e-5]}]
    grid_obj = GridSearchCV(clf, params, cv = cv, scoring = scoring_fun)
    grid_obj = grid_obj.fit(fatures_train, labels_train)
    return grid_obj.best_estimator_
