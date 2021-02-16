
# Importing required packages - 
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Reading in the data for modeling - 
data_for_model = pickle.load(open("abc_regression_modelling.p", "rb"))
X = data_for_model.drop("customer_loyalty_score", axis = 1)
y = data_for_model["customer_loyalty_score"]

# Dividing the data into train-test split -
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

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

# ~~~~~~~~~~~~ Using Grid search to find the best parameters for decision trees and random forest -

# DECISION TREE -

# Defining the grid search parameters for decision tree -
dt_gscv = GridSearchCV(estimator = DecisionTreeRegressor(random_state = 42),
                    param_grid = {"max_depth" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
                                  "min_samples_leaf" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                    cv = 5, 
                    scoring = "r2", 
                    n_jobs = -1)

# Fitting the grid search to get the best model -
dt_gscv.fit(X_train, y_train)

# Assessing the best score - 
dt_gscv.best_score_

# Saving the best parameters of the model -
dt_best_param = dt_gscv.best_params_

# RANDOM FOREST -

# Defining the grid search parameters for random forest -
rf_gscv = GridSearchCV(estimator = RandomForestRegressor(random_state = 42),
                    param_grid = {"n_estimators" : list(range(100,1000,50)),
                                  "max_depth" : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
                    cv = 5, 
                    scoring = "r2", 
                    n_jobs = -1)

# Fitting the grid search to get the best model -
rf_gscv.fit(X_train, y_train)

# Assessing the best score - 
rf_gscv.best_score_

# Saving the best parameters of the model -
rf_best_param = rf_gscv.best_params_

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
                        ("regressor", RandomForestRegressor(n_estimators = rf_best_param['n_estimators'],
                                                            max_depth = rf_best_param['max_depth'], random_state=42))])

# getting the data in the original state - 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fitting the model -
pipeline_regressor.fit(X_train, y_train)

# Saving the joblib file for using in the streamlit app -
import joblib
joblib.dump(pipeline_regressor, "pipeline_regressor.joblib")
