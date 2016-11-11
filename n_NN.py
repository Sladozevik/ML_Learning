# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 11:36:56 2016

@author: aslado
"""
from sklearn.neighbors import KNeighborsClassifier
from prep_terrain_data import makeTerrainData
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

X_train, y_train, X_test, y_test = makeTerrainData()

def n_nn(X_train, y_train, X_test, y_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train) 
    pred_nnn = neigh.predict(X_test)
#    print pred_nnn
    print('n nn Accuracy is: ', accuracy_score(y_test, pred_nnn))

def adaboost(X_train, y_train, X_test, y_test):
    ad_boo = AdaBoostClassifier()
    ad_boo.fit(X_train, y_train)
    pred_ad_boo = ad_boo.predict(X_test)
    print('AdaBoost Accurcy is: ', accuracy_score(y_test, pred_ad_boo))

def random_forest(X_train, y_train, X_test, y_test):
    ran_for = RandomForestClassifier()
    ran_for.fit(X_train, y_train)
    pred_ran_for = ran_for.predict(X_test)
    print('Random Fores Accuracy is: ', accuracy_score(y_test, pred_ran_for))
    
n_nn(X_train, y_train, X_test, y_test)
adaboost(X_train, y_train, X_test, y_test)
random_forest(X_train, y_train, X_test, y_test)