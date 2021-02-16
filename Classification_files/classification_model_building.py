
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, auc, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Loading the data -
data_for_model = pickle.load(open("abc_classification_modelling.p", "rb"))

# Breaking data into features and target -
X = data_for_model.drop("signup_flag", axis = 1)
y = data_for_model["signup_flag"]

# Spliting the data into test and train -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# Specifying the categorical features -
categorical_features = ["gender"]

# Initializing the one hot encoder -
one_hot_encoder = OneHotEncoder(sparse = False, drop = "first") 

# Runnig one hot encoding on train and test data separetly -
X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_features])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_features]) # Note - this is only transform and not fit_transform.

# Getting feature names to store in the df -
encoder_feature_names = one_hot_encoder.get_feature_names(categorical_features)

# Concating the one-hot-encoded variables and removing the old categorical variables from the train data -
X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop = True), X_train_encoded.reset_index(drop = True)], axis = 1)
X_train.drop(categorical_features, axis = 1, inplace = True)

# Concating the one-hot-encoded variables and removing the old categorical variables from the test data -
X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop = True), X_test_encoded.reset_index(drop = True)], axis = 1)
X_test.drop(categorical_features, axis = 1, inplace = True)

# Saving the objects for interactive trial for user - 
pickle.dump([X_train, X_test, y_train, y_test], open("user_trial_inputs.p", "wb")) 

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Trying the models first - 

# LOGISTIC REGRESSION -

# Fiting the logistic regression and making prediction -
clf = LogisticRegression(random_state = 42, max_iter = 1000)
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)

# We can get the probability instead of 0 and 1 using - 
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Creating confusion matrix -
conf_matrix = confusion_matrix(y_test, y_pred_class)

# Plot to show confusion matrix - 
plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for(i, j), corr_value in np.ndenumerate(conf_matrix) :
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

# Accuracy 
accuracy_score(y_test, y_pred_class)

# F1-Score 
f1_score(y_test, y_pred_class)

# AUC 
roc_auc_score(y_test, y_pred_prob)

# Making AUC -
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotting the AUC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# DECISION TREE

# Fiting the Decision Tree Classifier and making prediction -
clf = DecisionTreeClassifier(random_state = 42)
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)

# We can get the probability instead of 0 and 1 using - 
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Creating confusion matrix -
conf_matrix = confusion_matrix(y_test, y_pred_class)

# Plot to show confusion matrix - 
plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for(i, j), corr_value in np.ndenumerate(conf_matrix) :
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

# Accuracy 
accuracy_score(y_test, y_pred_class)

# F1-Score 
f1_score(y_test, y_pred_class)

# AUC 
roc_auc_score(y_test, y_pred_prob)

# Making AUC -
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotting the AUC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# RANDOM FOREST -

# Fiting the Random Forest Classifier and making prediction -
clf = RandomForestClassifier(random_state = 42)
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)

# We can get the probability instead of 0 and 1 using - 
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Creating confusion matrix -
conf_matrix = confusion_matrix(y_test, y_pred_class)

# Plot to show confusion matrix - 
plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for(i, j), corr_value in np.ndenumerate(conf_matrix) :
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

# Accuracy 
accuracy_score(y_test, y_pred_class)

# F1-Score 
f1_score(y_test, y_pred_class)

# AUC 
roc_auc_score(y_test, y_pred_prob)

# Making AUC -
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotting the AUC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# KNN -

# Fiting the KNN Classifier and making prediction -
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)

# We can get the probability instead of 0 and 1 using - 
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Creating confusion matrix -
conf_matrix = confusion_matrix(y_test, y_pred_class)

# Plot to show confusion matrix - 
plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap = "coolwarm")
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for(i, j), corr_value in np.ndenumerate(conf_matrix) :
    plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
plt.show()

# Accuracy 
accuracy_score(y_test, y_pred_class)

# F1-Score 
f1_score(y_test, y_pred_class)

# AUC 
roc_auc_score(y_test, y_pred_prob)

# Making AUC -
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plotting the AUC
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# ~~~~~~~~~~~~ Using Grid search to find the best parameters for decision trees and random forest -

# Since we get the best AUC from random forest and highest accuracy, we optimize the model using hyperparameter tuning -
# Defining the grid search parameters for random forest -
rf_gscv = GridSearchCV(estimator = RandomForestClassifier(random_state = 42),
                    param_grid = {"n_estimators" : list(range(100,1000,50)),
                                  "max_depth" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                    cv = 5, 
                    scoring = "roc_auc", 
                    n_jobs = -1)

# Fitting the grid search to get the best model -
rf_gscv.fit(X_train, y_train)

# Assessing the best score - 
rf_gscv.best_score_

# Saving the best parameters of the model -
rf_best_param = rf_gscv.best_params_

# We also find hyper-parameter for the decision tree to get best parameters to set default value in the application for user -
# Defining the grid search parameters for decision tree -
dt_gscv = GridSearchCV(estimator = DecisionTreeClassifier(random_state = 42),
                    param_grid = {"max_depth" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
                                  "min_samples_leaf" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                    cv = 5, 
                    scoring = "roc_auc", 
                    n_jobs = -1)

# Fitting the grid search to get the best model -
dt_gscv.fit(X_train, y_train)

# Assessing the best score - 
dt_gscv.best_score_

# Saving the best parameters of the model -
dt_best_param = dt_gscv.best_params_

# Saving the best parameters to set the default values in slider in streamlit -
pickle.dump([rf_best_param, dt_best_param], open("tuned_params.p", "wb"))

# ~~~~~~~~~~~~~~ Creating a pipeline to get prediction -

# Creating pipeline for the categorical features - 
categorical_transformer = Pipeline(steps = [("ohe", OneHotEncoder(handle_unknown = "ignore"))])

# Creating the column transformer to apply it to the new data - 
preprocessing_pipeline = ColumnTransformer(transformers = [("categorical", categorical_transformer, categorical_features)],
                                           remainder='passthrough')

# getting the tuned rf model from the grid search -
rf_regressor = rf_gscv.best_estimator_ 

# Instantiating the pipeline with the best parameters of the model -
pipeline_regressor = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("classifier", RandomForestClassifier(n_estimators = rf_best_param['n_estimators'],
                                                            max_depth = rf_best_param['max_depth'], random_state=42))])

# getting the data in the original state - 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fitting the model -
pipeline_regressor.fit(X_train, y_train)

# Saving the joblib file for using in the streamlit app -
import joblib
joblib.dump(pipeline_regressor, "pipeline_classifier.joblib")





