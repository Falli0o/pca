
# coding: utf-8

# In[ ]:


#use python 2.7
from __future__ import division
#get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 



# In[ ]:


dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')


# In[ ]:


XTrain = dataTrain[:, : -1]
YTrain = dataTrain[:, -1]
XTest = dataTest[:, : -1]
YTest = dataTest[:, -1]


# In[ ]:


def KNN_with_k(xtrain_data, ytrain_data, xtest_data, ytest_data, k):
    classify_results = []
    threshold = (k+1)/2 
    for eachrow in xtest_data:
        test_matrix = np.tile(eachrow, xtrain_data.shape[0])
        test_matrix = test_matrix.reshape(-1,13)
        for_cal_dis = np.subtract(test_matrix, xtrain_data)
        dis_matrix = np.dot(for_cal_dis, for_cal_dis.T)
        #try to find the diagonal
        distance = np.diag(dis_matrix)
        #print distance[np.argsort(distance)[0]]
        all_neighbors = [ytrain_data[i] for i in np.argsort(distance)[:k]]
        if np.count_nonzero(all_neighbors) >= threshold:
            result_label = 1
            classify_results.append(result_label)
        else:
            result_label = 0
            classify_results.append(result_label)
    accuracy_score = np.sum(classify_results == ytest_data)/ len(ytest_data)
    error_score = np.sum(classify_results != ytest_data)/ len(ytest_data)
    return (accuracy_score,error_score, classify_results)


# In[ ]:


#Exercise 1 (Nearest neighbor classification).
#Implementation I
KNN_with_k(XTrain, YTrain, XTrain, YTrain, 1)[:2] #the accuracy and error score by only using train data set


# In[ ]:


KNN_with_k(XTrain, YTrain, XTest, YTest, 1)[:2] #get the accuracy score and error score applied by the test data


# In[ ]:


#Exercise 1 (Nearest neighbor classification).
#Implementation II
k_1 = KNeighborsClassifier(n_neighbors=1)
k_1.fit(XTrain, YTrain)
predicted_results = k_1.predict(XTest)
from sklearn.metrics import accuracy_score
sklearn_acc_k_1 = accuracy_score(YTest, predicted_results)
sklearn_acc_k_1


# In[ ]:


def k_fold_split(Xdata, Ydata, fold_number, level):
    x_split_results = np.split(Xdata, fold_number)
    y_split_results = np.split(Ydata, fold_number)
    kf_xtest = x_split_results[level-1]
    kf_ytest = y_split_results[level-1]
    if level == 1:
        kf_xtrain = x_split_results[level:]
        kf_ytrain = y_split_results[level:]
    elif level == fold_number:
        kf_xtrain = x_split_results[:level-1]
        kf_ytrain = y_split_results[:level-1]
    else:
        kf_xtrain_part1 = x_split_results[:level-1]
        kf_xtrain_part2 = x_split_results[level:]
        kf_ytrain_part1 = y_split_results[:level-1]
        kf_ytrain_part2 = y_split_results[level:]
        kf_xtrain = np.concatenate((kf_xtrain_part1, kf_xtrain_part2))
        kf_ytrain = np.concatenate((kf_ytrain_part1, kf_ytrain_part2))
    y_size = int(((Ydata.shape[0])/fold_number)*(fold_number-1))
    kf_xtrain = np.array(kf_xtrain).reshape(-1,Xdata.shape[1])
    kf_ytrain = np.array(kf_ytrain).reshape(y_size)
    return (kf_xtrain, kf_ytrain, kf_xtest, kf_ytest)


# In[ ]:


def apply_data(XTrain, YTrain, fold_number, level, k):
    xtrain_data = k_fold_split(XTrain, YTrain, fold_number, level)[0]
    ytrain_data = k_fold_split(XTrain, YTrain, fold_number, level)[1]
    xtest_data = k_fold_split(XTrain, YTrain, fold_number, level)[2]
    ytest_data = k_fold_split(XTrain, YTrain, fold_number, level)[3]
    return KNN_with_k(xtrain_data, ytrain_data, xtest_data, ytest_data, k)


# In[ ]:


apply_data(XTrain, YTrain, 5, 1, 3)[:2]


# In[ ]:


def get_k_acc_err(Xdata, Ydata, fold_number, k):
    accuracies = []
    errors = []
    for i in range(fold_number):
        re = apply_data(Xdata, Ydata, fold_number, i+1, k)
        acc_cross = re[0]
        error_cross = re[1]
        accuracies.append(acc_cross)
        errors.append(error_cross)
    return (np.mean(accuracies), np.mean(errors))


# In[ ]:


acc_err_for_different_k = {}
for k in [1,3,5,7,9,11]:
    acc_err_for_different_k[k]=get_k_acc_err(XTrain, YTrain, 5, k)
acc_err_for_different_k


# In[ ]:


#Exercise 2 (Cross-validation).
#Implementation I
k_best = sorted(acc_err_for_different_k.items(), key = lambda x:x[1][1])[0][0]
k_best


# In[ ]:


#Exercise 2 (Cross-validation).
#Implementation II by sklearn
#The code is copied from assignment2 document.
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
cv = KFold(n_splits = 5)
all_acc_err_sklearn = {}
for k in [1, 3, 5, 7, 9, 11]:
    acc_results = []
    for train, test in cv.split(XTrain):
        XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train], XTrain[test], YTrain[train], YTrain[test]
        #estimate the performance
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(XTrainCV, YTrainCV)
        predicted_results_CV = neigh.predict(XTestCV)
        acc_score = accuracy_score(YTestCV, predicted_results_CV)
        acc_results.append(acc_score)
    acc = np.mean(acc_results)
    err = 1-acc
    all_acc_err_sklearn[k] = (acc, err)
all_acc_err_sklearn


# In[ ]:

plt.figure(1)
plt.scatter([1, 3, 5, 7, 9, 11], [i[1][1] for i in all_acc_err_sklearn.items()]);
plt.ylabel("Error Score");
plt.xlabel("k");
plt.title("Error score with different k");
plt.savefig("Error score with different k");


# In[ ]:


sklearn_k_best = sorted(all_acc_err_sklearn.items(), key = lambda x:x[1][1])[0][0]
sklearn_k_best


# In[ ]:


#Exercise 3 (Evaluation of classification performance).
#Implementation I
#estimate the performance
KNN_with_k(XTrain, YTrain, XTrain, YTrain, k_best)[:2] #for training
KNN_with_k(XTrain, YTrain, XTest, YTest, k_best)[:2] #for test


# In[ ]:


#Exercise 3 (Evaluation of classification performance).
#Implementation II by sklearn
#estimate the performance
choose_k = KNeighborsClassifier(n_neighbors=sklearn_k_best)
choose_k.fit(XTrain, YTrain)
choose_results = choose_k.predict(XTest)
choose_acc_score = accuracy_score(YTest, choose_results)
choose_err_score = 1 - choose_acc_score
choose_acc_score
choose_err_score


# In[ ]:


#Exercise 4
#Data normalization
#Implementation I
def normalization(train_data, transform_data):
    mean_row = np.mean(train_data, axis = 0)
    std_row = np.std(train_data, axis = 0)
    mean_matrix = np.tile(mean_row, transform_data.shape[0]).reshape(transform_data.shape[0],-1)
    #std_list = [(1/i) for i in std_row]
    std_list = 1/np.array(std_row)
    std_matrix = np.tile(std_list, transform_data.shape[0]).reshape(transform_data.shape[0],-1)
    dif_matrix = np.subtract(transform_data, mean_matrix)
    std_matrix_diag = np.diag(np.diag(std_matrix))
    normalizated_matrix = np.dot(dif_matrix, std_matrix_diag)
    return normalizated_matrix


# In[ ]:


nor_xtrain = normalization(XTrain, XTrain)
nor_xtest = normalization(XTrain, XTest)


# In[ ]:


#select k by using the function defined above
acc_err_for_different_k_nor = {}
for k in [1,3,5,7,9,11]:
    acc_err_for_different_k_nor[k]=get_k_acc_err(nor_xtrain, YTrain, 5, k)
acc_err_for_different_k_nor
#k-best is still 3.


# In[ ]:

plt.figure(2)
plt.scatter([1, 3, 5, 7, 9, 11], [i[1][1] for i in acc_err_for_different_k_nor.items()]);
plt.ylabel("Error Score");
plt.xlabel("k");
plt.title("Error score with different k after normalization");
plt.savefig("Error score with different k after normalization");


# In[ ]:


#select k
k_best_nor = sorted(acc_err_for_different_k_nor.items(), key = lambda x:x[1][1])[0][0]
k_best_nor


# In[ ]:


#estimate the performance
KNN_with_k(nor_xtrain, YTrain, nor_xtrain, YTrain, k_best_nor)[:2] #for training


# In[ ]:


#estimate the performance
KNN_with_k(nor_xtrain, YTrain, nor_xtest, YTest, k_best)[:2] #for test


# In[ ]:


#Implementation II by sklearn
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(XTrain)
XTrainN = scaler.transform(XTrain)
XTestN = scaler.transform(XTest)


# In[ ]:


#estimate the performance
normal = KNeighborsClassifier(n_neighbors = k_best)
normal.fit(XTrainN, YTrain)
predicted_results_normal = normal.predict(XTestN)
from sklearn.metrics import accuracy_score
sklearn_acc_normal_test = accuracy_score(YTest, predicted_results_normal) # for the independent test set
sklearn_err_normal_test = 1 - sklearn_acc_normal_test
sklearn_acc_normal_test
sklearn_acc_normal_test


# In[ ]:


sklearn_acc_normal_train = accuracy_score(YTrain, normal.predict(XTrainN))
sklearn_err_normal_train = 1- sklearn_acc_normal_train #for the training set
sklearn_acc_normal_train
sklearn_err_normal_train
#0.028000000000000025


# In[ ]:


#Discussion
#Draw data points
#draw the scatter plot to show the distribution of each feature
#the bule ponits stand for the label 0; otherwise the orange one stand for the label 1
def draw_plot(xtrain, ytrain):
    new_data_with_labels = np.hstack((xtrain,ytrain.reshape(-1,1)))
    label_0 = (new_data_with_labels[:, -1] == 0)
    data_label_0 = new_data_with_labels[label_0][:,:-1]
    label_1 = (new_data_with_labels[:, -1] == 1)
    data_label_1 = new_data_with_labels[label_1][:,:-1]
    x_cor = range(data_label_0.shape[1])
    for i in range(data_label_0.shape[0]):
        plt.scatter(x_cor, data_label_0[i], color = "blue")
    for i in range(data_label_1.shape[0]):
        plt.scatter(x_cor, data_label_1[i], color = "orange")


# In[ ]:

plt.figure(3)
draw_plot(XTrain, YTrain)
plt.ylabel("scaling");
plt.xlabel("feature");
plt.title("Distribution of each feature before normalization");
plt.savefig("Distribution of each feature before normalization");


# In[ ]:

plt.figure(4)
draw_plot(nor_xtrain, YTrain)
plt.ylabel("scaling");
plt.xlabel("feature");
plt.title("Distribution of each feature after normalization");
plt.savefig("Distribution of each feature after normalization");

