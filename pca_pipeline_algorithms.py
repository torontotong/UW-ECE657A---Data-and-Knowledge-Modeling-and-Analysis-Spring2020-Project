import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

from utility_functions import image_size

testSize = 0.3
randomState = np.random.RandomState(42)

# Define a pipeline to search for the best combination of PCA truncation
# and classifier regularization.
def pca_pipeline_classifiers(inputs, targets, n_components, algorithm='svm'):
    inputs_array = np.array(inputs, dtype=np.int)
    targets_array = np.array(targets, dtype=np.int)
    # split data to be training data and test data
    #    targets_array = label_binarize(targets_array, classes=[0, 1, 2])
    X_digits, X_test, y_digits, y_test = train_test_split(inputs_array, targets_array, test_size=testSize,
                                                        random_state=randomState)

    pca = PCA()
    # set the tolerance to a large value to make the example faster
    if algorithm == 'svm':
        svc = svm.SVC(random_state=randomState)
        pipe = Pipeline(steps=[('pca', pca), ('svm', svc)])
        param_grid = {
            'pca__n_components': [ 1 , 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
            #'svm__C': np.logspace(-2, 2, 2),
            #'logistic__C': np.logspace(-4, 4, 4),
        }
    elif algorithm == 'logistic regression':
        logistic = LogisticRegression(random_state=42,max_iter=100, tol=0.1)
        pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])
        param_grid = {
            'pca__n_components': [ 1 , 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
            #'logistic__C': np.logspace(-2, 2, 2),
        }
    elif algorithm == 'AdaBoost':
        clf = AdaBoostClassifier(n_estimators=100, random_state=42)
        pipe = Pipeline(steps=[('pca', pca), ('Adaboost', clf)])
        param_grid = {
            'pca__n_components': [ 1 , 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200],
            #'decisiontree__C': np.logspace(-2, 2, 2),
        }
    else:
        return 0
    search = GridSearchCV(pipe, param_grid, n_jobs=-1)
    search.fit(X_digits, y_digits)
    print("Best parameter (CV score={0:}: Best Components={1}".format( search.best_score_,search.best_params_))
    # Plot the PCA spectrum
    pca.fit(X_digits)

    fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
    ax0.plot(np.arange(1, pca.n_components_ + 1),
             pca.explained_variance_ratio_, '+', linewidth=2)
    ax0.set_ylabel('PCA explained variance ratio')

    ax0.axvline(search.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='n_components chosen')
    ax0.legend(prop=dict(size=12))

    # For each number of components, find the best classifier results
    results = pd.DataFrame(search.cv_results_)
    components_col = 'param_pca__n_components'
    best_clfs = results.groupby(components_col).apply(
        lambda g: g.nlargest(1, 'mean_test_score'))

    best_clfs.plot(x=components_col, y='mean_test_score', yerr='std_test_score',
                   legend=False, ax=ax1)
    ax1.set_ylabel(algorithm+' Classification accuracy')
    ax1.set_xlabel('n_components')

    plt.xlim(-1, n_components)

    plt.tight_layout()
    plt.savefig('pca-pipeline-'+algorithm+'-'+str(image_size)+'.png')
    plt.close()
    return search.best_estimator_.named_steps['pca'].n_components