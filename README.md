Download Link: https://assignmentchef.com/product/solved-data201-assignment-4
<br>
<h1>Dataset    ¶</h1>

The dataset was adapted from the Wine Quality Dataset

<a href="https://archive.ics.uci.edu/ml/datasets/Wine+Quality">(</a><a href="https://archive.ics.uci.edu/ml/datasets/Wine+Quality">https://archive.ics.uci.edu/ml/datasets/Wine+Quality </a><a href="https://archive.ics.uci.edu/ml/datasets/Wine+Quality">(</a><a href="https://archive.ics.uci.edu/ml/datasets/Wine+Quality">https://archive.ics.uci.edu/ml/datasets/Wine+Quality)</a><a href="https://archive.ics.uci.edu/ml/datasets/Wine+Quality">)</a>

<strong>Attribute Information:</strong>

<a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">For more information, read [Cortez et al., 2009: </a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">http://dx.doi.or</a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">g</a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">/10.1016</a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">/j</a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">.dss.2009.05.016 (http://dx.doi.or</a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">g</a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">/10.1016</a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">/j</a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">.dss.2009.05.016)</a><a href="https://dx.doi.org/10.1016/j.dss.2009.05.016">].</a>

Input variables (based on physicochemical tests):

<ul>

 <li>– fixed acidity</li>

 <li>– volatile acidity</li>

 <li>– citric acid</li>

 <li>– residual sugar</li>

 <li>– chlorides</li>

 <li>– free sulfur dioxide</li>

 <li>– total sulfur dioxide</li>

 <li>– density</li>

 <li>– pH</li>

 <li>– sulphates</li>

 <li>– alcohol</li>

</ul>

Output variable (based on sensory data):

<ul>

 <li>– quality (0: normal wine, 1: good wine)</li>

</ul>

<h1>Problem statement</h1>

Predict the quality of a wine given its input variables. Use AUC (area under the receiver operating characteristic curve) as the evaluation metric.

First, let’s load and explore the dataset.

In [1]:

In [2]:

Out[2]:

<strong>fixed_acidity    volatile_acidity    citric_acid    residual_sugar     chlorides    free_sulfur_dioxide    tot</strong>

<ul>

 <li>0 0.27        0.36        20.7        0.045      45.0</li>

 <li>3 0.30        0.34        1.6          0.049      14.0</li>

 <li>1 0.28        0.40        6.9          0.050      30.0</li>

 <li>2 0.23        0.32        8.5          0.058      47.0</li>

 <li>2 0.23        0.32        8.5          0.058      47.0</li>

</ul>

In [3]:

&lt;class ‘pandas.core.frame.DataFrame’&gt; RangeIndex: 4715 entries, 0 to 4714

Data columns (total 12 columns): fixed_acidity           4715 non-null float64 volatile_acidity        4715 non-null float64 citric_acid             4715 non-null float64 residual_sugar          4715 non-null float64 chlorides               4715 non-null float64 free_sulfur_dioxide     4715 non-null float64 total_sulfur_dioxide    4715 non-null float64 density                 4715 non-null float64 pH                      4715 non-null float64 sulphates               4715 non-null float64 alcohol                 4715 non-null float64 quality                 4715 non-null int64 dtypes: float64(11), int64(1) memory usage: 442.2 KB

In [4]:

Out[4]:

<ul>

 <li>3655</li>

 <li>1060</li>

</ul>

Name: quality, dtype: int64

Please note that this dataset is unbalanced.

<h1>Questions and Code</h1>

<strong>[1]. Split the given data using stratify sampling into 2 subsets: training (80%) and test (20%) sets. Use random_state = 42. [1 points] </strong>In [5]:

<strong>[2]. Use </strong><strong>GridSearchCV</strong><strong> and </strong><strong>Pipeline</strong><strong> to tune hyper-parameters for 3 different classifiers including </strong>

<strong>KNeighborsClassifier </strong><strong>, </strong><strong>LogisticRegression</strong><strong> and </strong><strong>svm.SVC</strong><strong> and report the corresponding AUC values on the training and test sets. Note that a scaler may need to be inserted into each pipeline. [6 points]</strong>

Hint: You may want to use kernel=’rbf’ and tune C and gamma for svm.SVC . Find out how to enable probability estimates (for Question 3).

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">Document: </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">https://scikit-learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">enerated/sklearn.svm.SVC.html#sklearn.svm.SVC</a>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">(https://scikit-learn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC">enerated/sklearn.svm.SVC.html#sklearn.svm.SVC)</a>

In [12]:

K-Nearest Neighbors best parameters: {‘clf__n_neighbors’: 45, ‘clf__p’: 1}

K-Nearest Neighbors AUC score(training set): 1.0

K-Nearest Neighbors AUC score(test set): 0.9349366337144774

K-Nearest Neighbors Confusion Matrix(training set):

[[2924    0]

[   0  848]]

K-Nearest Neighbors Confusion Matrix(test set):

[[701  30]

[ 66 146]] time: 0.13440759579340616

Logistic Regression best parameters: {‘clf__C’: 100, ‘clf__penalty’: ‘l1’}

Logistic Regression AUC score(training set): 0.7867747883488629 Logistic Regression AUC score(test set): 0.7987184781767029

Logistic Regression Confusion Matrix(training set): [[2754  170]

[ 605  243]]

Logistic Regression Confusion Matrix(test set):

[[690  41]

[158  54]] time: 0.03498464822769165

SVC best parameters: {‘clf__C’: 1, ‘clf__gamma’: 100}

SVC AUC score(training set): 0.9991603321890405 SVC AUC score(test set): 0.9088480499703171

SVC Confusion Matrix(training set):

[[2918    6]

[  43  805]]

SVC Confusion Matrix(test set):

[[718  13]

[112 100]] time: 0.6369452118873596

<strong>[3]. Train a soft </strong><strong>VotingClassifier</strong><strong> with the estimators are the three tuned pipelines obtained from [2]. Report the AUC values on the training and test sets. Comment on the performance of the ensemble model. [1 point]</strong>

Hint: consider the voting method.

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">Document: </a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">https://scikitlearn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">enerated/sklearn.ensemble.Votin</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">Classifier.html#sklearn.ensemble.Votin</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">Classifier</a>

<a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">(https://scikitlearn.or</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">/stable/modules/</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">enerated/sklearn.ensemble.Votin</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">Classifier.html#sklearn.ensemble.Votin</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">g</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">Classifier</a><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier">)</a>

In [13]:

start = time.time()

ensemble = VotingClassifier(estimators=pipelines, voting=’soft’, n_jobs=-1).fit(X_train

, y_train)

ensemble_train = roc_auc_score(y_train, ensemble.predict_proba(X_train)[:,1], average=

‘macro’)

ensemble_test = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:,1], average=’mac ro’)

print(“VotingClassifier AUC score(training set): <strong>{}</strong>“.format(ensemble_train)) print(“VotingClassifier AUC score(test set): <strong>{}</strong>“.format(ensemble_test))

print(“VotingClassifier Confusion Matrix(training set):<strong>
</strong> <strong>{}</strong>“.format(confusion_matrix(y

_train, ensemble.predict(X_train))))

print(“VotingClassifier Confusion Matrix(test set):<strong>
</strong> <strong>{}</strong>“.format(confusion_matrix(y_tes t, ensemble.predict(X_test))))

end = time.time() print(“time: <strong>{}</strong><strong>
</strong>“.format((end-start)/60))

VotingClassifier AUC score(training set): 0.9999903208321503 VotingClassifier AUC score(test set): 0.9399956121105748 VotingClassifier Confusion Matrix(training set):

[[2923    1]

[   8  840]] VotingClassifier Confusion Matrix(test set):

[[709  22]

[ 84 128]] time: 0.691833249727885

<strong>The ensemble model performs marginally better than K-Nearest Neighbors(the difference is 0.005 so might as well be the same performance), slightly better than SVC and significantly better than logistic regression. The ensemble model doesn’t improve on the best performing estimator (KNN) in any meaningful way</strong>

<strong>[4]. Redo [3] with a sensible set of </strong><strong>weights</strong><strong> for the estimators. Comment on the performance of the ensemble model in this case. [1 point]</strong>

In [14]:

start = time.time() weight_params = []

<strong>for</strong> w1 <strong>in</strong> range(1,4):

<strong>for</strong> w2 <strong>in</strong> range(1,4):

<strong>for</strong> w3 <strong>in</strong> range(1,4):            weight_params.append([w1, w2, w3])

ensemble_weighted = VotingClassifier(estimators=pipelines, voting=’soft’, n_jobs=-1) ensemble_gs = GridSearchCV(ensemble_weighted, param_grid={‘weights’: weight_params}, n_ jobs=-1, cv=3, scoring=’roc_auc’)

ensemble_fit = ensemble_gs.fit(X_train, y_train)

weighted_train = ensemble_fit.score(X_train, y_train)<em>#roc_auc_score(y_train, ensemble_w eighted.predict_proba(X_train)[:,1], average=’macro’)</em>

weighted_test = ensemble_fit.score(X_test, y_test)<em>#roc_auc_score(y_test, ensemble_weigh ted.predict_proba(X_test)[:,1], average=’macro’)</em>

print(“VotingClassifier best weights: <strong>{}</strong>“.format(ensemble_fit.best_params_)) print(“VotingClassifier(weights=<strong>{}</strong>) AUC score(training set): <strong>{}</strong>“.format(ensemble_fit.be st_params_[‘weights’], weighted_train))

print(“VotingClassifier(weights=<strong>{}</strong>) AUC score(test set): <strong>{}</strong>“.format(ensemble_fit.best_p arams_[‘weights’], weighted_test))

print(“VotingClassifier(weights=<strong>{}</strong>) Confusion Matrix(training set):<strong>
</strong> <strong>{}</strong>“.format(ensemb le_fit.best_params_[‘weights’], confusion_matrix(y_train, ensemble_fit.predict(X_train

))))

print(“VotingClassifier(weights=<strong>{}</strong>) Confusion Matrix(test set):<strong>
</strong> <strong>{}</strong>“.format(ensemble_f it.best_params_[‘weights’], confusion_matrix(y_test, ensemble_fit.predict(X_test))))

end = time.time() print(“time: <strong>{}</strong><strong>
</strong>“.format((end-start)/60))

VotingClassifier best weights: {‘weights’: [2, 1, 1]}

VotingClassifier(weights=[2, 1, 1]) AUC score(training set): 1.0

VotingClassifier(weights=[2, 1, 1]) AUC score(test set): 0.941073226131172 1

VotingClassifier(weights=[2, 1, 1]) Confusion Matrix(training set):

[[2924    0]

[   0  848]]

VotingClassifier(weights=[2, 1, 1]) Confusion Matrix(test set): [[710  21]

[ 76 136]] time: 24.4167094151179

<strong>KNN got a perfect 100% accuracy on the training set and the highest AUC score for the test set. It makes it sensible to have a weight of 2 for KNN and 1 for the others. I also tested it out via</strong>

<strong>GridSearchCV and it also gave me 2,1,1 as the best parameters. Giving KNN more voting power gave us a 100% on the training set that we didn’t get from the unweighted Voting Classifier. It also gives us a better AUC score for the test set.</strong>

<strong>[5]. Use the </strong><strong>VotingClassifier</strong><strong> with </strong><strong>GridSearchCV</strong><strong> to tune the hyper-parameters of the individual estimators. The parameter grid should be a combination of those in [2]. Report the AUC values on the training and test sets. Comment on the performance of the ensemble model. [1 point] </strong>Note that it may take a long time to run your code for this question.

<a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">Document: </a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">https://scikit-learn.or</a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">g</a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">/stable/modules/ensemble.html#usin</a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">g</a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">-the-votin</a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">g</a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">classifier-with-</a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">g</a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">ridsearchcv </a><a href="https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv">(https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv)</a>

In [9]:

start = time.time()

params = {} estimators = [] <strong>for</strong> name, classifier, param <strong>in</strong> zip(names, classifiers, parameters):

estimators.append((name, classifier))    <strong>for</strong> k <strong>in</strong> param:        params[k.replace(‘clf’, ‘vote__’+name)] = param[k]

vot_ = VotingClassifier(estimators=estimators, voting=’soft’, n_jobs=-1) pipe=Pipeline(steps=[(‘scale’, scaler), (‘vote’, vot_)])

gs_clf_cv = GridSearchCV(estimator=pipe, param_grid=params, cv=3, n_jobs=-1, scoring=’r oc_auc’)

clf_cv = gs_clf_cv.fit(X_train, y_train) cv_train_score = clf_cv.score(X_train, y_train) cv_test_score = clf_cv.score(X_test, y_test)

print(“VotingClassifier with GridSearchCV best parameters: <strong>{}</strong>“.format(clf_cv.best_param s_))

print(“VotingClassifier with GridSearchCV AUC score(training set): <strong>{}</strong>“.format(cv_train_ score))

print(“VotingClassifier with GridSearchCV AUC score(test set): <strong>{}</strong>“.format(cv_test_score

))

print(“VotingClassifier with GridSearchCV Confusion Matrix(training set):<strong>
</strong> <strong>{}</strong>“.format( confusion_matrix(y_train, clf_cv.predict(X_train))))

print(“VotingClassifier with GridSearchCV Confusion Matrix(test set):<strong>
</strong> <strong>{}</strong>“.format(conf usion_matrix(y_test, clf_cv.predict(X_test))))

end = time.time() print(“time: <strong>{}</strong><strong>
</strong>“.format((end-start)/60))

VotingClassifier with GridSearchCV best parameters: {‘vote__K-Nearest Neig hbors__n_neighbors’: 70, ‘vote__K-Nearest Neighbors__p’: 2, ‘vote__Logisti c Regression__C’: 1000, ‘vote__Logistic Regression__penalty’: ‘l1’, ‘vote_ _SVC__C’: 1, ‘vote__SVC__gamma’: 100}

VotingClassifier with GridSearchCV AUC score(training set): 0.999991127429 471

VotingClassifier with GridSearchCV AUC score(test set): 0.9399633482177426 VotingClassifier with GridSearchCV Confusion Matrix(training set):

[[2923    1]

[   8  840]]

VotingClassifier with GridSearchCV Confusion Matrix(test set): [[715  16]

[ 88 124]] time: 107.44066168467204 <strong>Imagine taking 100 minutes to execute and still getting a lower score than the previous two Voting Classifiers. The base Voting Classifier, Voting Classifier with GridSearchCV and SVC all yielded incredibly similiar results whilst the Voting Classifier with estimator weights of 2,1,1 seem to pull ahead by a whopping 0.001 for the test set!</strong>

In [ ]: