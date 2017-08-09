# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 16:52:03 2015

@author: fisheryzhq
"""

import random
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score    


#Histogram (data: y-axis value, x: x-axis value)
def ShowHistogram(data, x, x_lable, y_lable, title):
    
    plt.bar(x, data)
    plt.title(title)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.show()

def ShowConfusionMatrix(m, method):
    
    plt.matshow(m)
    plt.title('Confusion matrix of ' + method)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

#PCA
def PerformPCA(train, test, components):
    
    train.index = range(0, len(train))
    test.index = range(0, len(test))
    #val.index = range(0, len(val))
    
    pca = PCA(n_components = components)
    pca.fit(train.ix[:,0:(len(train.columns) - 2)])
    NewTrain = pca.transform(train.ix[:,0:(len(train.columns) - 2)])
    NewTest = pca.transform(test.ix[:,0:(len(test.columns) - 2)])
    #NewVal = pca.transform(val.ix[:,0:(len(val.columns) - 2)])
    
    NewTrain = pd.DataFrame(NewTrain)
    NewTest = pd.DataFrame(NewTest)
    #NewVal = pd.DataFrame(NewVal)

    NewTrain['target'] = train.ix[:,len(train.columns) - 1]
    NewTest['target'] = test.ix[:,len(test.columns) - 1]
    #NewVal['target'] = val.ix[:,len(val.columns) - 1]
    
    return NewTrain, NewTest    
    
#KNN
def KNNClassifier(train, test, num_k):
    
    correct = 0
    wrong = 0
    pred = list()
    
    temp_train = train
    
    temp_train.index = range(0, len(train))
    
    X = np.array(train.ix[:, 0:(len(temp_train.columns) - 2)])
        
    tree = KDTree(X, leaf_size = 2)
    
    for index, row in test.iterrows():
                
        Y = np.array(test.ix[index, 0:(len(test.columns)-2)])
        
        dist, inds = tree.query(Y.reshape(1, -1), k=num_k)
        
        count_zero = 0
        count_one = 0        
        
        for ind in inds[0]:
            if temp_train.ix[ind, len(temp_train.columns)-1] == 0:
                count_zero += 1
            else:
                count_one += 1
    
        if count_one > count_zero and test.ix[index, len(test.columns)-1] == 1:
            correct += 1
        elif count_zero > count_one and test.ix[index, len(test.columns)-1] == 0:
            correct += 1
        else:
            wrong += 1
        
        if count_one >= count_zero:
            pred.append(1)
        else:
            pred.append(0)
    return float(correct)/(correct + wrong), pred

#Naive Bayes Classifier which take two params: training dataset and testing dataset
def NBClassifier(train, test):
    
    mean_pos = list()
    mean_neg = list()
    var_pos = list()
    var_neg = list()
    pred = list()
    
    #train.index = range(0, len(train))
    #test.index = range(0, len(test))
    
    for i in range(0, len(train.columns) - 2):
        
        mean_pos.append(np.average(train[train.ix[:,len(train.columns) - 1] == 1].ix[:,i]))  
        var_pos.append(np.var(train[train.ix[:,len(train.columns) - 1] == 1].ix[:,i]))
        
        mean_neg.append(np.average(train[train.ix[:,len(train.columns) - 1] == 0].ix[:,i]))  
        var_neg.append(np.var(train[train.ix[:,len(train.columns) - 1] == 0].ix[:,i]))        
        
    
    #print var_pos
    #prior
    lenx1 = len(train[train.ix[:,len(train.columns) - 1] == 1])
    lenx2 = len(train[train.ix[:,len(train.columns) - 1] == 0])
    
    prior1 = float(lenx1)/(lenx1 + lenx2)
    prior2 = float(lenx2)/(lenx1 + lenx2)

   
    #traning and testing
    correct = 0
    wrong = 0
    
    for index, row in test.iterrows(): 
        
        #c1_likehood = list()
        #c2_likehood = list()
        post1 = 1
        post2 = 1        
        
        for i in range(0, len(test.columns)-2):
            c1_likehood = (np.exp(-np.square(test.ix[index, i] - mean_pos[i])/(2*mean_pos[i]))/np.sqrt(var_pos[i]))
            c2_likehood = (np.exp(-np.square(test.ix[index, i] - mean_neg[i])/(2*mean_neg[i]))/np.sqrt(var_neg[i]))
            
            if var_pos[i] == 0:
                c1_likehood = 1
            if var_neg[i] == 0:
                c2_likehood = 1
            post1 = post1 * c1_likehood
            post2 = post2 * c2_likehood
            
        post1 = post1 * prior1
        post2 = post2 * prior2
        
        #print post1
        #print post2
        
        if post1 > post2 and test.ix[index, len(test.columns)-1] == 1:
            correct += 1
        elif post1 < post2 and test.ix[index, len(test.columns)-1] == 0 :
            correct += 1
        else:
            wrong += 1
            
        if post1 >= post2:
            pred.append(1)
        else:
            pred.append(0)
    #print float(correct)/(correct + wrong)
    return float(correct)/(correct + wrong), pred

#Loading data
dataset_origin = open('spambase.data')

dataset = list()

for line in dataset_origin:
    
    dataset.append(map(float, line.strip().split(',')))


EmailSpam = pd.DataFrame(dataset)


#Generate training and testing
#Spliting the data into 50%-70% training and the rest is testing
is_test = np.random.uniform(1, 0, len(EmailSpam)) <= random.uniform(0.3, 0.4)
train = EmailSpam[is_test == False]
test = EmailSpam[is_test == True]

##Validation
#is_val = np.random.uniform(1, 0, len(train)) <= random.uniform(0.2, 0.3)
#train_val = train[is_val == False]
#val = train[is_val == True]

#Find the new base for the dataset
pca = PCA()
pca.fit(train.ix[:,0:56])
print "The proportion of variance of each of the principal components is: ", pca.explained_variance_ratio_
ShowHistogram(pca.explained_variance_ratio_, np.arange(1, 58), 'components', 'proportion of variance', 'PCA without standardization')

#The first three components are most important, since they contain more than
#99% information of the dataset. We can set the components to 3
print pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1] + pca.explained_variance_ratio_[2]


#http://blog.explainmydata.com/2012/07/should-you-apply-pca-to-your-data.html
#NewTrain, NewTest, NewVal = PerformPCA(train, test, val, 3)

#Stardardization of raw data
from sklearn.preprocessing import StandardScaler

train_std = StandardScaler().fit_transform(train.ix[:,0:56])
train_std = pd.DataFrame(train_std)
train_std['target'] = train.ix[:,57]
test_std = StandardScaler().fit_transform(test.ix[:,0:56])
test_std = pd.DataFrame(test_std)
test_std['target'] = test.ix[:,57]

pca = PCA()
pca.fit(train_std.ix[:,0:56])
print "The proportion of variance of each of the principal components is: ", pca.explained_variance_ratio_
ShowHistogram(pca.explained_variance_ratio_, np.arange(1, 58), 'components', 'proportion of variance', 'PCA after standardization')

acc_knn = list()
acc_knn_pca = list()
acc_knn_std = list()
cm_knn = [[0, 0],[0, 0]]
cm_knn_pca = [[0, 0],[0, 0]]
cm_knn_std = [[0, 0],[0, 0]]
acc_nb = list()
acc_nb_pca = list()
acc_nb_std = list()
cm_nb = [[0, 0],[0, 0]]
cm_nb_pca = [[0, 0],[0, 0]]
cm_nb_std = [[0, 0],[0, 0]]
acc_svm = list()
acc_svm_pca = list()
acc_svm_std = list()
cm_svm = [[0, 0],[0, 0]]
cm_svm_pca = [[0, 0],[0, 0]]
cm_svm_std = [[0, 0],[0, 0]]
acc_rf = list()
acc_rf_pca = list()
acc_rf_std = list()
cm_rf = [[0, 0],[0, 0]]
cm_rf_pca = [[0, 0],[0, 0]]
cm_rf_std = [[0, 0],[0, 0]]

for i in range(0, 10):
    
    print "This is the ", i, "run!"
    
    #Split the data into train and test
    is_test = np.random.uniform(1, 0, len(EmailSpam)) <= random.uniform(0.3, 0.4)
    train = EmailSpam[is_test == False]
    test = EmailSpam[is_test == True]
    
    NewTrain, NewTest = PerformPCA(train, test, 3)
    
    train_std = StandardScaler().fit_transform(train.ix[:,0:56])
    train_std = pd.DataFrame(train_std)
    train_std['target'] = train.ix[:,57]
    test_std = StandardScaler().fit_transform(test.ix[:,0:56])
    test_std = pd.DataFrame(test_std)
    test_std['target'] = test.ix[:,57]
    NewTrain_std, NewTest_std = PerformPCA(train_std, test_std, 43)    
    
    #KNN 
    acc, pred_knn = KNNClassifier(train, test, 1)
    acc_knn.append(acc)
    cm_knn = cm_knn + confusion_matrix(NewTest['target'], pred_knn)
    acc, pred_knn = KNNClassifier(NewTrain, NewTest, 1)
    acc_knn_pca.append(acc)
    cm_knn_pca = cm_knn_pca + confusion_matrix(NewTest['target'], pred_knn)
    acc, pred_knn = KNNClassifier(NewTrain_std, NewTest_std, 1)
    acc_knn_std.append(acc)
    cm_knn_std = cm_knn_std + confusion_matrix(NewTest['target'], pred_knn)
    
    #NB
    acc, pred_nb = NBClassifier(train, test)
    acc_nb.append(acc)
    cm_nb = cm_nb + confusion_matrix(NewTest['target'], pred_nb)
    acc, pred_nb = NBClassifier(NewTrain, NewTest)
    acc_nb_pca.append(acc)
    cm_nb_pca = cm_nb_pca + confusion_matrix(NewTest['target'], pred_nb)
    acc, pred_nb = NBClassifier(NewTrain_std, NewTest_std)
    acc_nb_std.append(acc)
    cm_nb_std = cm_nb_std + confusion_matrix(NewTest['target'], pred_nb)
    
    #SVM
    clf_SVC = SVC(probability = True)
    clf_SVC.fit(train.ix[:,0:56], train.ix[:,57])
    pred_SVC = clf_SVC.predict(test.ix[:,0:56])
    cm_svm = cm_svm + confusion_matrix(NewTest['target'], pred_SVC)
    temp = pd.crosstab(test.ix[:,57], pred_SVC)
    acc = float(temp.ix[0,0] + temp.ix[1, 1]) / (temp.ix[0,0] + temp.ix[0, 1] + temp.ix[1,0] + temp.ix[1, 1])
    acc_svm.append(acc)
    
    clf_SVC = SVC(probability = True)
    clf_SVC.fit(NewTrain.ix[:,0:3], NewTrain.ix[:,3])
    pred_SVC = clf_SVC.predict(NewTest.ix[:,0:3])
    cm_svm_pca = cm_svm_pca + confusion_matrix(NewTest['target'], pred_SVC)
    temp = pd.crosstab(NewTest.ix[:,3], pred_SVC)
    acc = float(temp.ix[0,0] + temp.ix[1, 1]) / (temp.ix[0,0] + temp.ix[0, 1] + temp.ix[1,0] + temp.ix[1, 1])
    acc_svm_pca.append(acc)
    
    clf_SVC.fit(NewTrain_std.ix[:,0:43], NewTrain_std.ix[:,43])
    pred_SVC = clf_SVC.predict(NewTest_std.ix[:,0:43])
    cm_svm_std = cm_svm_std + confusion_matrix(NewTest['target'], pred_SVC)
    temp = pd.crosstab(NewTest_std.ix[:,43], pred_SVC)
    acc = float(temp.ix[0,0] + temp.ix[1, 1]) / (temp.ix[0,0] + temp.ix[0, 1] + temp.ix[1,0] + temp.ix[1, 1])
    acc_svm_std.append(acc)  
    
    #RF
    clf_RF = RandomForestClassifier(n_estimators=25)

    clf_RF.fit(train.ix[:,0:56], train.ix[:,57])
    pred_RF = clf_RF.predict(test.ix[:,0:56])
    cm_rf = cm_rf + confusion_matrix(NewTest['target'], pred_RF)
    temp = pd.crosstab(test.ix[:,57], pred_RF)

    acc = float(temp.ix[0,0] + temp.ix[1, 1]) / (temp.ix[0,0] + temp.ix[0, 1] + temp.ix[1,0] + temp.ix[1, 1])
    acc_rf.append(acc)
    
    clf_RF = RandomForestClassifier(n_estimators=25)

    clf_RF.fit(NewTrain.ix[:,0:3], NewTrain.ix[:,3])
    pred_RF = clf_RF.predict(NewTest.ix[:,0:3])
    cm_rf_pca = cm_rf_pca + confusion_matrix(NewTest['target'], pred_RF)
    temp = pd.crosstab(NewTest.ix[:,3], pred_RF)

    acc = float(temp.ix[0,0] + temp.ix[1, 1]) / (temp.ix[0,0] + temp.ix[0, 1] + temp.ix[1,0] + temp.ix[1, 1])
    acc_rf_pca.append(acc)

    clf_RF.fit(NewTrain_std.ix[:,0:43], NewTrain_std.ix[:,43])
    pred_RF = clf_RF.predict(NewTest_std.ix[:,0:43])
    cm_rf_std = cm_rf_std + confusion_matrix(NewTest['target'], pred_RF)
    temp = pd.crosstab(NewTest_std.ix[:,43], pred_RF)
    
    acc = float(temp.ix[0,0] + temp.ix[1, 1]) / (temp.ix[0,0] + temp.ix[0, 1] + temp.ix[1,0] + temp.ix[1, 1])
    acc_rf_std.append(acc)


print acc_knn
print "The average accuracy of KNN with raw data: ", np.average(acc_knn)
print acc_knn_pca
print "The average accuracy of KNN without standardization: ", np.average(acc_knn_pca)
print acc_knn_std
print "The average accuracy of KNN with standardization: ", np.average(acc_knn_std)

print acc_nb
print "The average accuracy of NB with raw data: ", np.average(acc_nb)
print acc_nb_pca
print "The average accuracy of NB without standardization: ", np.average(acc_nb_pca)
print acc_nb_std
print "The average accuracy of NB with standardization: ", np.average(acc_nb_std)

print acc_svm
print "The average accuracy of SVM with raw data: ", np.average(acc_svm)
print acc_svm_pca
print "The average accuracy of SVM without standardization: ", np.average(acc_svm_pca)
print acc_svm_std
print "The average accuracy of SVM with standardization: ", np.average(acc_svm_std)

print acc_rf
print "The average accuracy of RF with raw data: ", np.average(acc_rf)
print acc_rf_pca
print "The average accuracy of RF without standardization: ", np.average(acc_rf_pca)
print acc_rf_std
print "The average accuracy of RF with standardization: ", np.average(acc_rf_std)

#plot the average accuracy
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

width = 0.15
ind = np.arange(3)

y = [np.average(acc_knn), np.average(acc_knn_pca), np.average(acc_knn_std)]
rects1 = ax.bar(ind, y, width, color='r', align='center')
y = [np.average(acc_nb), np.average(acc_nb_pca), np.average(acc_nb_std)]
rects2 = ax.bar(ind+width, y, width, color='b', align='center')
y = [np.average(acc_svm), np.average(acc_svm_pca), np.average(acc_svm_std)]
rects3 = ax.bar(ind+width*2, y, width, color='g', align='center')
y = [np.average(acc_rf), np.average(acc_rf_pca), np.average(acc_rf_std)]
rects4 = ax.bar(ind+width*3, y, width, color='y', align='center')

ax.set_ylabel('Accuracy')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('Raw Data', 'PCA', 'PCA_STD') )
ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('KNN', 'NB', 'SVM', 'RF') )

plt.title('The average accuracy of four classifier with or without PCA')
plt.show()

#show confusion matrix
ShowConfusionMatrix(cm_knn, 'KNN with Raw Data')
print "The precision of KNN with raw data is ", float(cm_knn[0][0])/(cm_knn[0][0] + cm_knn[0][1])
print "The recall of KNN with raw data is ", float(cm_knn[0][0])/(cm_knn[0][0] + cm_knn[1][0])
ShowConfusionMatrix(cm_knn_pca, 'KNN with PCA(NO STD)')
print "The precision of KNN with raw data is ", float(cm_knn_pca[0][0])/(cm_knn_pca[0][0] + cm_knn_pca[0][1])
print "The recall of KNN with raw data is ", float(cm_knn_pca[0][0])/(cm_knn_pca[0][0] + cm_knn_pca[1][0])
ShowConfusionMatrix(cm_knn_std, 'KNN wiht PCA(STD)')
print "The precision of KNN with raw data is ", float(cm_knn_std[0][0])/(cm_knn_std[0][0] + cm_knn_std[0][1])
print "The recall of KNN with raw data is ", float(cm_knn_std[0][0])/(cm_knn_std[0][0] + cm_knn_std[1][0])
ShowConfusionMatrix(cm_nb, 'NB with Raw Data')
print "The precision of NB with raw data is ", float(cm_nb[0][0])/(cm_nb[0][0] + cm_nb[0][1])
print "The recall of NB with raw data is ", float(cm_nb[0][0])/(cm_nb[0][0] + cm_nb[1][0])
ShowConfusionMatrix(cm_nb_pca, 'NB with PCA(NO STD)')
print "The precision of NB with raw data is ", float(cm_nb_pca[0][0])/(cm_nb_pca[0][0] + cm_nb_pca[0][1])
print "The recall of NB with raw data is ", float(cm_nb_pca[0][0])/(cm_nb_pca[0][0] + cm_nb_pca[1][0])
ShowConfusionMatrix(cm_nb_std, 'NB wiht PCA(STD)')
print "The precision of NB with raw data is ", float(cm_nb_std[0][0])/(cm_nb_std[0][0] + cm_nb_std[0][1])
print "The recall of NB with raw data is ", float(cm_nb_std[0][0])/(cm_nb_std[0][0] + cm_nb_std[1][0])
ShowConfusionMatrix(cm_svm, 'SVM with Raw Data')
print "The precision of SVM with raw data is ", float(cm_svm[0][0])/(cm_svm[0][0] + cm_svm[0][1])
print "The recall of SVM with raw data is ", float(cm_svm[0][0])/(cm_svm[0][0] + cm_svm[1][0])
ShowConfusionMatrix(cm_svm_pca, 'SVM with PCA(NO STD)')
print "The precision of SVM with raw data is ", float(cm_svm_pca[0][0])/(cm_svm_pca[0][0] + cm_svm_pca[0][1])
print "The recall of SVM with raw data is ", float(cm_svm_pca[0][0])/(cm_svm_pca[0][0] + cm_svm_pca[1][0])
ShowConfusionMatrix(cm_svm_std, 'SVM wiht PCA(STD)')
print "The precision of SVM with raw data is ", float(cm_svm_std[0][0])/(cm_svm_std[0][0] + cm_svm_std[0][1])
print "The recall of SVM with raw data is ", float(cm_svm_std[0][0])/(cm_svm_std[0][0] + cm_svm_std[1][0])
ShowConfusionMatrix(cm_rf, 'RF with Raw Data')
print "The precision of RF with raw data is ", float(cm_rf[0][0])/(cm_rf[0][0] + cm_rf[0][1])
print "The recall of RF with raw data is ", float(cm_rf[0][0])/(cm_rf[0][0] + cm_rf[1][0])
ShowConfusionMatrix(cm_rf_pca, 'RF with PCA(NO STD)')
print "The precision of RF with raw data is ", float(cm_rf_pca[0][0])/(cm_rf_pca[0][0] + cm_rf_pca[0][1])
print "The recall of RF with raw data is ", float(cm_rf_pca[0][0])/(cm_rf_pca[0][0] + cm_rf_pca[1][0])
ShowConfusionMatrix(cm_rf_std, 'RF wiht PCA(STD)')
print "The precision of RF with raw data is ", float(cm_rf_std[0][0])/(cm_rf_std[0][0] + cm_rf_std[0][1])
print "The recall of RF with raw data is ", float(cm_rf_std[0][0])/(cm_rf_std[0][0] + cm_rf_std[1][0])


#Experiment with RF
weighted_RF = RandomForestClassifier(n_estimators=25)
clf_RF.fit(NewTrain_std.ix[:,0:43], NewTrain_std.ix[:,43])
posterior = clf_RF.predict_proba(NewTest_std.ix[:,0:43])
thresh = np.arange(0.5, 1.02, 0.02)
fp = []
acc = []

for t in thresh:
    pred_rf = list()

    for pair in posterior:
        if pair[1] >= t:
            pred_rf.append(1)
        else:
            pred_rf.append(0)
    
    con_ma = confusion_matrix(NewTest['target'], pred_rf)
    fp.append(con_ma[0][1])
    acc.append(accuracy_score(NewTest['target'], pred_rf))

    print
    print "with posterior threshold of spam >=", t
    print confusion_matrix(NewTest['target'], pred_rf)
    print accuracy_score(NewTest['target'], pred_rf)

plt.clf()
plt.figure()
plt.plot(thresh, fp)
plt.xlabel('The posterior threshold of spam email')
plt.ylabel('Number of False Positives')
plt.title('False Positive')
plt.show()

plt.clf()
plt.figure()
plt.plot(thresh, acc)
plt.xlabel('The posterior threshold of spam email')
plt.ylabel('Percentage of Accuracy')
plt.title('Accuracy')
plt.show()

fp = []
acc = []
x = np.arange(1, 20)

for i in range(1, 20):
    wei_svm = SVC(class_weight = {0:i, 1:1 })
    #clf_SVC.fit(NewTrain_std.ix[:,0:43], NewTrain_std.ix[:,43])
    wei_svm.fit(NewTrain_std.ix[:,0:43], NewTrain_std.ix[:,43])
    #pred_svm = clf_SVC.predict(NewTest_std.ix[:,0:43])
    pred_wsvm = wei_svm.predict(NewTest_std.ix[:,0:43])
    print
    print "with non-spam class weight =", i
    #print confusion_matrix(NewTest['target'], pred_svm)
    #print accuracy_score(NewTest['target'], pred_svm)
    t = confusion_matrix(NewTest['target'], pred_wsvm)
    fp.append(t[0][1])
    print "True Positives:", t[0][0]
    print "False Positives:", t[0][1]
    print "False Negatives:", t[1][0]
    print "True Negatives:", t[1][1]
    print accuracy_score(NewTest['target'], pred_wsvm)
    acc.append(accuracy_score(NewTest['target'], pred_wsvm))

plt.clf()
plt.figure()
plt.plot(x, fp)
plt.xlabel('Non-spam class weight')
plt.ylabel('Number of False Positives')
plt.title('False Positive')
plt.show()

plt.clf()
plt.figure()
plt.plot(x, acc)
plt.xlabel('Non-spam class weight')
plt.ylabel('Percentage of Accuracy')
plt.title('Accuracy')
plt.show()



