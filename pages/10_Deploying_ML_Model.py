
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ AB TESTING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~ Importing the packages ~~~~~~~~~~~~~~~~~~~~

import streamlit as st
from pages.common_functions.app_functions import title_image, plot_image_121

# Getting the title image - 
title_image()

# Title - 
st.markdown("""
            # Deploying ML Models using FastAPI
            
            In this project we go through the steps to setup API's for our models using FastAPI to assist us in making real time predictions.

            """, unsafe_allow_html=True)

plot_image_121("fastapi_ml", "ML_Deployment", [1.8,3,1])

st.markdown("""

## Table of contents

- [00. Project Overview](#overview-main)
- [Context](#overview-context)
- [Actions](#overview-actions)
- [Results](#overview-results)
- [01. FastAPI Overview](#fastapi-overview)
- [02. Creating Basic API](#basic-api)
- [03. Building Our Model](#building-ml-model)
- [04. The Request Body](#request-body)
- [05. The Endpoint](#endpoint)
- [06. Growth & Next Steps](#next-steps)
___


## Project Overview  <a name="overview-main"></a>

### Context <a name="overview-context"></a>

In the project, we assisted the ABC grocery in creating 3 different machine learning models - 
1. Predicting Customer Loyalty Scores
2. Enhancing Customer Targeting
3. Image Classification

While the models work well, it has no value to the business till the users can use it. This is where model deployment comes into play. Deploying ML models as an API service enables the business to use the model at will and get the most value out of it. In this section we will walk-through on take the model from *jupyter notebook* and deploy it online.

### Actions <a name="overview-actions"></a>

Deployment is usually the last step in any Data Science Project lifecycle, to be able to integrate your ML/DL model into a web application is quite an important task. The first step in deploying the model is ensuring data consistency. What it really means is, that when we get the raw data, we have to ensure that we apply the same pre-processing steps that we did before training the data. 

The data should be in the exact format as required by the model including handling missing values, categorical variables, feature scaling, feature selection, and feature engineering steps. This can be achieved easily by creating a pipeline to make all the changes. A pipeline acts like a black box that takes in the raw data and returns clean and formatted data which can be consumed by our model.

The next step is to save the final trained model as a pickle object so that it can be accessed easily along with the created pipeline. We can then load the pickle object in our API and use the predict method to get the prediction. 

### Results <a name="overview-results"></a>

The goal of this project is to wrap the model in an API. The API loads the pipeline and the saved model to make the prediction. The input to the API is sent as a JSON object. FastAPI can also be used along with the 'pydantic' package which has a module 'BaseModel' which helps to define the input being sent into the API. 

The API then returns a similar JSON object with the prediction. We can parse the JSON output to get the results for the users in the desired format. This API can be hosted on any platform of our choice so that it can be accessed as a micro-service to get real-time predictions.

## FastAPI Overview <a name="fastapi-overview"></a>

API is short for Application Programming Interface. It’s simply an intermediary between two independent applications that communicate with each other. If you’re a developer and want to make your application available for other developers to use and integrate with, you build an API that acts as an entry point to your app. The developers will therefore have to communicate with this API through HTTP requests to consume and interact with your service.

FastAPI is currently the go-to framework for building robust and high-performance APIs that scale in production environments. FastAPI has gained a lot of popularity lately and saw a huge increase in user adoption among web developers but also data scientists and ML engineers.

**The most interesting features of FastAPI are -**

**1. A Simple Syntax:**
FastAPI’s syntax is simple and this makes it fast to use. It actually resembles Flask’s syntax: so if you’re thinking about migrating from Flask to FastAPI, the transition should be easy.

**2. A blazing fast framework:**
According to tech empower, an independent website that benchmarks web servers by running a variety of tests on them, FastAPI+uvicorn is one of the fastest web servers.

**3. Asynchronous request:**
Asynchronous programming is a pattern of programming that enables code to run separately from the main application thread. Asynchronous programming is used in many use-cases such as event-driven systems, highly scalable apps, and I/O-bound tasks such as reading and writing files through the network.

**4. Validation of string query parameters:**
FastAPI allows validating user inputs by adding constraints on the string and numerical query parameters.

**5. Better error handling and custom messages:**
With FastAPI, you can define and raise custom errors with a specific message and status code. This helps other developers easily understand the errors and debug while using your API.

## Creating Basic API <a name="fastapi-overview"></a>

Before creating our ML model let's start by creating a basic API that’s going to return us a simple message. With just one line of code, we can initialize a FastAPI instance. This app object is responsible for handling the requests for our REST API. 

```python

# importing packages -
from fastapi import FastAPI
import uvicorn

# declaring our FastAPI instance -
app = FastAPI()

```

Now that we have a FastAPI app object, we can use it to define the output for a simple get request as demonstrated below. The get request above for the root URL simply returns a JSON output with a welcome message.

```python

# defining path operation for root endpoint -
@app.get('/')
def main():
    return {'message': 'Welcome to homepage'}

```

## Building Our Model <a name="building-ml-model"></a>

For demonstration, we will train the model for 'Predicting Customer Loyalty Scores' and save the appropriate required objects (the pipeline object and the model) to make predictions using the API. 

```python

# Importing required packages - 
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib

# Reading in the data for modeling - 
data_for_model = pickle.load(open("abc_regression_modelling.p", "rb"))

# separating the features adn the target variable -
X = data_for_model.drop("customer_loyalty_score", axis = 1)
y = data_for_model["customer_loyalty_score"]

# ~~~~~~~~~~~~~~ Creating a pipeline to get prediction -

# Specifying the categorical features -
categorical_features = ["gender"]

# Creating pipeline for the categorical features - 
categorical_transformer = Pipeline(steps = [("ohe", OneHotEncoder(handle_unknown = "ignore"))])

# Creating the column transformer to apply it to the new data - 
preprocessing_pipeline = ColumnTransformer(transformers = [("categorical", categorical_transformer, categorical_features)],
                                           remainder='passthrough')

# Instantiating the pipeline with the best parameters of the model -
pipeline_regressor = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("regressor", RandomForestRegressor(n_estimators = 1000,
                                                            max_depth = 8, random_state=42))])

# getting the data in the original state - 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Fitting the model -
pipeline_regressor.fit(X_train, y_train)

# Saving the joblib file for using in the streamlit app -
joblib.dump(pipeline_regressor, "Resources/pipeline_regressor.joblib")

```

Now that we have our model ready we need to define the format of the data we are going to provide to our model to make the predictions.

## The Request Body <a name="request-body"></a>

The data sent from the client side to the API is called a request body. The data sent from API to the client is called a response body. 

Using this approach, one can pass the data from the client to our API. In FastAPI, to simplify things, we use Pydantic models to define the data structure for the receiving data. The Pydantic does all the type checking for the parameters and returns explainable errors if the wrong type of parameter is received. Let’s add a data class to our existing code and create a route for the request body:

    
```python

# importing the package - 
from pydantic import BaseModel

# defining the data types in the input class -
class Input(BaseModel):
    distance_from_store : float
    gender : str
    credit_score : float
    total_sales : int
    total_items : int
    transaction_count : int
    product_area_count : int    

```

## The Endpoint <a name="endpoint"></a>

Now that we have a request body all that’s left to do is to add an endpoint that’ll make the prediction and return it as a JSON response.

Although in the 'Basic API' section we used the GET method, while building APIs for ML models, we use the POST method. This is because it's considered better practice to send parameters in JSON rather than in URL.

We will create a “/get_loyalty_score" route which will take the data sent by the client request body and our API will return the response as a JSON object containing the result.

**Below are the steps that take place in the API -**

1. The pipeline model is unpickled and saved as 'pipeline_regressor'. This model object will be used to get the predictions.
2. The “/get_loyalty_score" route function declares a parameter called regressor_input of the "Input" Model type. This parameter can be accessed as a dictionary. The dictionary object will allow us to access the values of the parameters as key-value pairs.
3. The values sent by the client are saved in the required format. The values are now fed to the model predict function and we have our prediction for the data provided.

```python 

# importing the packages
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# initializing the app
app = FastAPI(title="Ritesh ML Models")

# defining the data types in the input class -
class Input(BaseModel):
    distance_from_store : float
    gender : str
    credit_score : float
    total_sales : int
    total_items : int
    transaction_count : int
    product_area_count : int    

# defining path operation for root endpoint -
@app.get("/")
def read_main():
    return {"message": "Welcome"}

# defining path operation for prediction endpoint -
@app.post('/get_loyalty_score/', tags = ['Predicting Customer Loyalty Scores'])
def customer_loyalty_scores(regressor_input : Input):
    
    # import the pipeline -
    pipeline_regressor = joblib.load("Resources/pipeline_regressor.joblib")
    
    # Converting input to dataframe -
    X_input = pd.DataFrame({"distance_from_store" : regressor_input.distance_from_store,
                            "gender" : regressor_input.gender,
                            "credit_score" : regressor_input.credit_score,
                            "total_sales" : regressor_input.total_sales,
                            "total_items" : regressor_input.total_items,
                            "transaction_count" : regressor_input.transaction_count,
                            "product_area_count" : regressor_input.product_area_count,
                            "average_basket_value" : regressor_input.total_sales/regressor_input.transaction_count}, index = [0])
    
    # Returning the prediction value - 
    return {"Loyalty Score" : pipeline_regressor.predict(X_input)[0]}

```

## Growth & Next Steps <a name="next-steps"></a>

In this section, we went through how we can use FastAPI to quickly deploy a machine learning model. FastAPI is a lightweight and fast framework that data scientists can use to create APIs for machine learning models that can easily be integrated into larger applications.

Now that we have a working API, we can easily deploy it anywhere as a Docker container. If you aren’t familiar with Docker, it is basically a tool that lets you package and run applications in isolated environments called containers. After dockerizing the API, we choose to deploy it on any platform of our choice like AWS, Azure, Google Cloud Platform, etc. 

This particular API and all APIs for this application are hosted on Heroku and can be accessed at - [API Deployment](https://ml-api-ritesh.herokuapp.com/docs)



""", unsafe_allow_html=True)