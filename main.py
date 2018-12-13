# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 12:16:59 2018

@author: nitesh.yadav
"""
import Student_Intervention as si
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def main():
     # load data from csv file
     full_data_frame, features, labels = si.Data_Load()
     # Explore data
     si.Data_Exploration(full_data_frame)
     # Prepare Data
     features, labels = si.Data_Preparation(full_data_frame)
     # preprocess features
     features = si.Preprocess_Features(features)
     # split dataset into training and testing
     features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = .25, random_state = 0)
     # Show the results of the split
     print("Training set has {} samples.".format(features_train.shape[0]))
     print("Testing set has {} samples.".format(features_test.shape[0]))
     # Initialize the three models
     clf_1 = DecisionTreeClassifier()
     clf_2 = KNeighborsClassifier(n_neighbors = 7)
     clf_3 = SVC()
     # Train and test above models
     si.Train_Predict(clf_1, features_train, features_test, labels_train, labels_test)
     si.Train_Predict(clf_2, features_train, features_test, labels_train, labels_test)
     si.Train_Predict(clf_3, features_train, features_test, labels_train, labels_test)
     # Tune the chosen model
     clf = si.Model_Tuning(clf_3, features_train, labels_train)
     print("Performance details of tuned classifier:")
     print("Accuracy for training set: {:.4f}.".format(si.Predict_Labels(clf, features_train, labels_train)))
     print("Accuracy for test set: {:.4f}.".format(si.Predict_Labels(clf, features_test, labels_test)))

if __name__ == "__main__":
    main()