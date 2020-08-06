import math

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, label_binarize, OneHotEncoder
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics, svm
import time
from sklearn.metrics import roc_curve

#from PCA_Reduce_Image_features import *
from sklearn.tree import DecisionTreeClassifier

from utility_functions import *
from pca_pipeline_algorithms import *

testSize = 0.1
randomState = np.random.RandomState(42)

def algorithm_func(inputs, targets, class_type_num, algorithm_f='svm'):

    inputs_array = np.array(inputs, dtype=np.int)
    targets_array = np.array(targets, dtype=np.int)
    # split data to be training data and test data
    X_train, X_test, y_train, y_test = train_test_split(inputs_array, targets_array, test_size=testSize, random_state=randomState)

    start_time = time.time()
    if algorithm_f == 'svm':
        clf = svm.SVC(random_state=randomState)
    elif algorithm_f == 'logistic regression':
        clf = LogisticRegression(random_state=42,max_iter=100, tol=0.1)
    elif algorithm_f == 'decision_tree':
        clf = DecisionTreeClassifier(random_state=42)
    elif algorithm_f == 'knn':
        clf = KNeighborsClassifier(n_neighbors=5)
    elif algorithm_f == 'GaussianNB':
        clf = GaussianNB()
    elif algorithm_f == 'AdaBoost':
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    else:
        clf = svm.SVC(random_state=randomState)

    clf = clf.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(X_test)
    print("Algorithm ran %s seconds"% (time.time() - start_time))
    fpr ,tpr , _ = roc_curve(y_test, y_pred, pos_label=2 , drop_intermediate=False)

    cm = get_confusion_matrix(y_test, y_pred, class_type_num)
    accuracy = total_accuracy(cm)
    precision = precision_average(cm)
    print( algorithm_f + ' classifier is Confusion Matrix \n Predicted vs Actual \n{0}'.format(cm))
    print( algorithm_f + ' classifier is accuracy {0}'.format(accuracy*100))
    print( algorithm_f + ' classifier is precision {0}'.format(precision*100))
    return fpr, tpr, clf

# PCA method function
def pca_func(d_array, n_components):
    print("Computing pca projection")
    t_start = time.time()
    pca = PCA(n_components=n_components, svd_solver='full')
    pca_proj = pca.fit_transform(d_array)
    print("Done in {0} sec.".format(time.time()-t_start))
    return pca_proj


def run_classification_with_PCA(TrainingDataMat, Label_lists ,n_components, TestDataMat, TestFileList):
    n_algorithm = 3
    algm_name = ['svm', 'logistic regression', 'AdaBoost']
    fpr = dict()
    tpr = dict()
    clf = {}
    best_n_components = {}
    lowDimensionData = {}
    lowTestMat = {}
    if n_components > TrainingDataMat.__len__():
        n_components = TrainingDataMat.__len__()
    lowDataMat = pca_func(TrainingDataMat, n_components)
    if n_components > TestDataMat.__len__():
        n_components = TestDataMat.__len__()
    lowTestDataMat = pca_func(TestDataMat, n_components)
    for i in range(n_algorithm):
        best_n_components[i] = pca_pipeline_classifiers(lowDataMat, Label_lists, n_components, algm_name[i])
        lowDimensionData[i] = pca_func(TrainingDataMat, best_n_components[i])
        lowTestMat[i] = pca_func(lowTestDataMat, best_n_components[i])
    print("Start to train data")
    start_time = time.time()
    for i in range(n_algorithm):
        fpr[i], tpr[i] ,clf[i] = algorithm_func(lowDimensionData[i], Label_lists, 2, algm_name[i])
        pred = clf[i].predict(lowTestMat[i])
        pca_test_result = dict(zip(TestFileList, pred))
        for item in sorted(pca_test_result.keys()):
            print('Image {0} is class {1}'.format(item, pca_test_result[item]))
    print("Image size {0}x{0} with PCA Total runtime {1} Seconds".format( image_size,time.time()-start_time))
    plot_roc(fpr, tpr, n_algorithm, 'pca')



def run_classification(TrainingDataMat, Label_lists, TestDataMat, TestFileList):
    n_algorithm = 3
    algm_name = ['svm', 'logistic regression', 'AdaBoost']
    fpr = dict()
    tpr = dict()
    clf = {}
    #best_n_components = {}

    start_time = time.time()
    for i in range(n_algorithm):
        fpr[i], tpr[i],clf[i]  = algorithm_func(TrainingDataMat, Label_lists, 2, algm_name[i])
        pred_full_dataset = clf[i].predict(TestDataMat)
        test_result = dict(zip(TestFileList, pred_full_dataset))
        for item in sorted(test_result.keys()):
            print('Image {0} is class {1}'.format(item, test_result[item]))

    plot_roc(fpr, tpr, n_algorithm, 'full_data')
    print("Image size {0}x{0}Total runtime {1} Seconds".format(image_size, time.time() - start_time))
    return


def main():
    # Load dataset
    TrainingDataMat, Label_lists ,n_components = loadDataset(train_dataset_folder)
    TestDataMat, TestFileList = loadTestDataset()

    #run_classification_with_PCA(TrainingDataMat, Label_lists ,n_components, TestDataMat, TestFileList)
    run_classification(TrainingDataMat, Label_lists, TestDataMat, TestFileList)

    return
if __name__ == '__main__':
    main()
