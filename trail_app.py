
# IMPORT IMPORTANT LIBRARIES

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score, accuracy_score, confusion_matrix, f1_score, auc, roc_curve, roc_auc_score
import joblib

# ~~~~~~~~~~~~~~~~~ ALL REQUIRED FUNCTIONS ~~~~~~~~~~~~~~~~~

# Function to create Analytics page - 
def analytics_Page():
    
    # Introduction of the page -
    st.write("""
             # Welcome to the Analytics Page!
             
             This page is designed to give an overview of the business to the stakeholders.  
             
             We use a tablaue dashboard to give the overview of the business. 
             """)
    st.write('---')
    
    st.write("## Tableau Dashboard")
    
    html_temp = """
                <div class='tableauPlaceholder' id='viz1613340201972' style='position: relative'><noscript><a href='#'><img alt=' ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;AB&#47;ABCGroceryDashboard&#47;Final&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='ABCGroceryDashboard&#47;Final' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;AB&#47;ABCGroceryDashboard&#47;Final&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en' /><param name='filter' value='publish=yes' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1613340201972');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1377px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
                """
    components.html(html_temp, width=1130, height=900)
    
# Function to get different parameters according to user selected model -
def add_parameter_ui(model, navigation_tab):
    
    if navigation_tab == 'Customer Loyalty Calculator':
        rf_best_param, dt_best_param = pickle.load(open("Regression_Files/tuned_params.p", "rb"))
    elif navigation_tab == 'Marketing Recommender':
        rf_best_param, dt_best_param = pickle.load(open("Classification_files/tuned_params.p", "rb"))
    
    # Empty list to save the parameters -
    params = dict()
    
    # Draw a line below the navigation selectbox -
    st.sidebar.write('---')
    
    # Depending on the regressor -
    if model != '~ Select a Model ~':
        # Heading for user -
        st.sidebar.markdown('#### You can choose different parameters for the model below - ')
        st.sidebar.write(' ')
        
    if model == 'Logistic Regression':
        # No parameters for logistic regression -
        st.sidebar.write('We only train simple Logistic Regression model.')
        
    if model == 'Linear Regression':
        # No parameters for linear regression -
        st.sidebar.write('We only train Ordinary Least Squares model.')
        
    if model == 'Decision Trees':
        # Slider for min leaf samples and max depth along with default values from the grid search -
        min_samples_leaf = st.sidebar.slider('Minimum leaf samples', 1, 10, value = dt_best_param['min_samples_leaf'])
        max_depth = st.sidebar.slider('Max_depth', 2, 10, value = dt_best_param['max_depth'])
        # Saving the parameters in the dictionary -
        params['min_samples_leaf'] = min_samples_leaf
        params['max_depth'] = max_depth
        
    if model == 'Random Forest':
        # Slider for n_estimators and max depth along with default values from the grid search -
        max_depth = st.sidebar.slider('Max_depth', 2, 15, value = rf_best_param['max_depth'])
        n_estimators = st.sidebar.slider('n_estimators', 1, 1000, value = rf_best_param['n_estimators'])
        # Saving the parameters in the dictionary -
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators
    return params

# Function to instantiate a user selected model with choosen parameters - 
# Fuction takes in a regresoor name and user entered parameters.
# It returns a regressor with the sppecified parameter, which can be used to train and predict on trial data for users -
def get_regressor(regressor_name, params):
    if regressor_name == 'Linear Regression':
        regressor = LinearRegression()
    elif regressor_name == 'Decision Trees':
        regressor = DecisionTreeRegressor(min_samples_leaf = params['min_samples_leaf'], max_depth = params['max_depth'], random_state = 123)
    elif regressor_name == 'Random Forest':
        regressor = RandomForestRegressor(n_estimators = params['n_estimators'], max_depth = params['max_depth'], random_state = 123)
    return regressor

# It returns a classifier with the sppecified parameter, which can be used to train and predict on trial data for users -
def get_classifier(classifier_name, params):
    if classifier_name == 'Logistic Regression':
        clf = LogisticRegression()
    elif classifier_name == 'Decision Trees':
        clf = DecisionTreeClassifier(min_samples_leaf = params['min_samples_leaf'], max_depth = params['max_depth'], random_state = 123)
    elif classifier_name == 'Random Forest':
        clf = RandomForestClassifier(n_estimators = params['n_estimators'], max_depth = params['max_depth'], random_state = 123)
    return clf

# Function to get prediction -
def prediction_input(section_expander, navigation_tab):
    
    # Setting a flag to ensure all correct inputs have been entered -
    input_check = [1,1,1,1,1,1]
    
    # ~~~~~~~~~~~~~ Getting in all the inputs and checking if they are as desired -
    
    # Dividing the window into 2 parts to get the input -
    input_col1, input_col2 = section_expander.beta_columns((1,1))
    
    # Getting distance from store - 
    distance_from_store = input_col1.text_input('Distance from store (in miles). Preferably between 0 and 5', value = 1.3)
    try:
        distance_from_store = float(distance_from_store)
        input_check[0] = 1
    except:
        input_check[0] = 0
        input_col1.info('Please enter a number.')
    
    # No check required as it is a select box -
    gender_M = input_col1.selectbox('Gender', ['M', 'F'])
    
    # Getting credit score between 0 and 1
    credit_score = input_col1.text_input('Credit Score (Between 0 and 1)', value = 0.5)
    try:
        credit_score = float(credit_score)
        if credit_score > 1 or credit_score < 0:
            input_check[1] = 0
            input_col1.info('Please enter a number between 0 and 1')
        else:
            input_check[1] = 1
    except:
        input_check[1] = 0
        input_col1.info('Please enter a number.')
    
    # Getting Total sales of the customer -
    total_sales = input_col1.text_input('Total Sales ($). Please enter numeric values', 100)
    try:
        total_sales = float(total_sales)
        input_check[2] = 1
    except:
        input_check[2] = 0
        input_col1.info('Please enter a number.')
    
    # Getting total items - 
    total_items = input_col2.text_input('Total items. Please enter numeric values', 10)
    try:
        total_items = int(float(total_items))
        input_check[3] = 1
    except:
        input_check[3] = 0
        input_col2.info('Please enter a number.')
        
    # Getting the product area name -
    product_area_count = input_col2.text_input('Product Area Count (between 1 and 5)', 3)
    try:
        product_area_count = int(float(product_area_count))
        if product_area_count > 5 or product_area_count < 1:
            input_check[4] = 0
            input_col2.info('Please enter a number between 1 and 5')
        else:
            input_check[4] = 1
    except:
        input_check[4] = 0
        input_col2.info('Please enter a number.')
    
    # getting transaction Count -
    transaction_count = input_col2.text_input('Transaction Count', 50)
    try:
        transaction_count = int(float(transaction_count))
        if transaction_count == 0:
            input_check[5] = 0
            input_col2.info('Please enter a number greater than 0.')
        else:
            input_check[5] = 1
    except:
        input_check[5] = 0
        input_col2.info('Please enter a number.')
    
    try:
        X_input = pd.DataFrame({"distance_from_store" : distance_from_store,
                                "gender" : gender_M,
                                "credit_score" : credit_score,
                                "total_sales" : total_sales,
                                "total_items" : total_items,
                                "transaction_count" : transaction_count,
                                "product_area_count" : product_area_count,
                                "average_basket_value" : total_sales/transaction_count}, index = [0])
    except:
        pass
    
    col1, col2, col3 = section_expander.beta_columns((1,1,1))
    
    if sum(input_check) != 6:
        col2.info(' Check all the inputs! ')
    else:
        if navigation_tab == 'Customer Loyalty Calculator':
            # Reading in the trained pipeline to make prediction -
            pipeline_regressor = joblib.load("Regression_Files/pipeline_regressor.joblib")
            prediction = pipeline_regressor.predict(X_input)[0]
            col2.write(f'**Customer Loyalty Score** = {round(prediction, 2)}')
        elif navigation_tab == 'Marketing Recommender':
            # Reading in the trained pipeline to make prediction -
            pipeline_clf = joblib.load("Classification_files/pipeline_classifier.joblib")
            prediction = pipeline_clf.predict_proba(X_input)[0][1]
            col2.write(f'**Probability of signing up** - {round(prediction, 3)}')

# Function to display Customer loyalty page -
def customer_loyalty_calculator():
    
    # Brief explanation of the section -
    
    st.write("""
             ## Welcome to the Customer Loyalty Score calculator!!
             
             #### How to navigate the page -
             
             This page contains 3 sections -
             * **About:** Here you can read about the main idea of the page.
             * **Calculate Customer Loyalty:** This section uses a tuned random forest to make prediction.
             * **Explore different machine-learning models:** This section is designed for users to play with small chunk of data and see how different models perform.
             
             """)
    
    st.write(' ')
    st.write(' ')
    
    # About Section -
    about_expander = st.beta_expander('About')
    about_expander.markdown("""
                
                The company had earlier hired a consulancy agency to calculate the customer loyalty score of existing customers. After sometime, they acquired new customers and wanted to calculate the customer loyalty score in-house in order to save the costs.  
                
                This app currently uses random forest model to predict the customer loyalty score after trying various models. The explore different models section is designed to show an interactive space which can be used to test different models. I only provide a few parameters to tune but this can be extended to other available parameters as well.
                
                """)
    
    st.write(' ')
    
    # Making expander for the getting prediction - 
    prediction_expander = st.beta_expander('Calculate Customer Loyalty Score')
    prediction_expander.write("""
                              Please enter the inputs below to get prediction (Dummy values have been entered to assist the user) -  
                              
                              """)
    prediction_input(prediction_expander, navigation_tab)
    
    st.write(' ')
    
    # Section to allow users to play with test and train data -
    model_expander = st.beta_expander('Explore different machine-learning models')
    model_expander.write("""
                              In this section you can play around with 3 different models and their respective parameters to see how a particular model responds.  
                              * **Decision tree** (min_leafs_sample, max_depth)  
                              * **Random Forest** (n_estimators, max_depth)  
                              * **Linear Regression** (No parameters available)     
                                             
                              """)
    model_expander.write(' ')
    
    
    X_train, X_test, y_train, y_test = pickle.load(open("Regression_Files/user_trial_inputs.p", "rb"))
    
    # Select box to select the type of model to play with -
    regressor_model = model_expander.selectbox('Choose a model', ['~ Select a Model ~', 'Decision Trees', 'Random Forest', 'Linear Regression'])
    
    # Depending on the selection - 
    if regressor_model != '~ Select a Model ~':    
        
        # Setting the sidebar to display parameters according to the model -
        params = add_parameter_ui(regressor_model, navigation_tab)
        regressor = get_regressor(regressor_model, params)
        
        # Training the model - 
        regressor.fit(X_train, y_train)
        pred_values = regressor.predict(X_test)
        
        # Printing the model R2 score -
        model_expander.write(f'* **R2-score** of the model - **{r2_score(y_test, pred_values)}**')
        
        model_expander.write(' ')
        
        # Dividing the screen into 2 parts to display the graphs -
        model_expander_col1, model_expander_col2 = model_expander.beta_columns((1,1))
        
        # Scatter plot of actual values vs the predicted values -
        plt.style.use("bmh")
        plt.scatter(y_test, pred_values)
        plt.plot([0, 1], [0, 1],'--', color = 'black')
        plt.xlabel('Test Values')
        plt.ylabel('Predicted Values')
        plt.title('Graph comparing predicted values to actual values')
        model_expander_col1.pyplot()
        mpl.rcParams.update(mpl.rcParamsDefault)
        
        # Depending on the model -
        # In case of Linear regression, we show the model coefficients - 
        if regressor_model == 'Linear Regression':
            # Saving the coeffiecients and columns names in a data frame -
            coefficients = pd.DataFrame(regressor.coef_)
            input_variable_names =  pd.DataFrame(X_train.columns)
            summary_stats = pd.concat([input_variable_names, coefficients], axis = 1)
            summary_stats.columns = ["input_variable", "coefficient"]
            # Displaying the coefficients dataframe -
            model_expander_col2.write('Regressor Coefficients - ')
            model_expander_col2.write(' ')
            model_expander_col2.dataframe(summary_stats)
            
        # In case of Decision tree or random forest, we display the feature importance -
        elif (regressor_model == 'Decision Trees' or regressor_model == 'Random Forest'):
            # Getting the feature importance and column names in a dataframe -
            feature_importance = pd.DataFrame(regressor.feature_importances_)
            feature_names = pd.DataFrame(X_train.columns)
            feature_importance_summary = pd.concat([feature_names, feature_importance], axis = 1)
            feature_importance_summary.columns = ["input_variable", "feature_importance"]
            feature_importance_summary.sort_values(by = "feature_importance", inplace = True)
            # Plotting a horizontal bar chart of the feature importance -
            plt.barh(feature_importance_summary["input_variable"], 
                     feature_importance_summary["feature_importance"],
                     color = 'forestgreen')
            plt.title("Feature Importance of Input Variables")
            plt.xlabel("Feature Importance")
            plt.tight_layout()
            model_expander_col2.pyplot()

def marketing_recommender():
    # Brief explanation of the section -
    
    st.write("""
             ## Welcome to the customer recommender area!
             
             #### How to navigate the page -
             
             This page contains 3 sections -
             * **About:** Here you can read about the main idea of the page.
             * **Calculate Probability of Customer Churn:** This section uses a tuned random forest to make prediction.
             * **Explore different machine-learning models:** This section is designed for users to play with small chunk of data and see how different models perform.
             
             """)
    
    st.write(' ')
    st.write(' ')
    
    # About Section -
    about_expander = st.beta_expander('About')
    about_expander.markdown("""
                
                In last 3 months, the marketing team ran a promotional campaign which would give customers free delivery for a $100 membership. We conducted AB testing to see whether a certain type of flyer would have higher sign-up rate. 
                We concluded that customers were more likely to sign-up if sent a particular flyer.  
                
                Later we designed a recommendation engine which would inform how likely a customer is to sign up for the promotion. This would help marketing team target the required customers and reduce the overall cost of the campaign.  
                
                This app currently uses random forest model to predict the recommendation.   
                
                The explore different models section is designed to show an interactive space which can be used to test different models. I only provide a few parameters to tune but this can be extended to other available parameters as well.
                
                """)
    
    st.write(' ')
    
    # Making expander for the getting prediction - 
    prediction_expander = st.beta_expander('Calculate Probability of Customer Churn')
    prediction_expander.write("""
                              Please enter the inputs below to get prediction (Dummy values have been entered to assist the user) -  
                              
                              """)
    prediction_input(prediction_expander, navigation_tab)
    
    st.write(' ')
    
    # Section to allow users to play with test and train data -
    model_expander = st.beta_expander('Explore different machine-learning models')
    model_expander.write("""
                              In this section you can play around with 3 different models and their respective parameters to see how a particular model responds.  
                              * **Decision tree** (min_leafs_sample, max_depth)  
                              * **Random Forest** (n_estimators, max_depth)  
                              * **Logistic Regression** (No parameters available)     
                                             
                              """)
    model_expander.write(' ')
    
    X_train, X_test, y_train, y_test = pickle.load(open("Classification_Files/user_trial_inputs.p", "rb"))
    
    # Select box to select the type of model to play with -
    clf_model = model_expander.selectbox('Choose a model', ['~ Select a Model ~', 'Decision Trees', 'Random Forest', 'Logistic Regression'])
    
    # Depending on the selection - 
    if clf_model != '~ Select a Model ~':    
        
        # Setting the sidebar to display parameters according to the model -
        params = add_parameter_ui(clf_model, navigation_tab)
        clf = get_classifier(clf_model, params)
        
        # Training the model - 
        clf.fit(X_train, y_train)
        
        # Making prediction -
        y_pred_class = clf.predict(X_test)

        # We can get the probability instead of 0 and 1 using - 
        y_pred_prob = clf.predict_proba(X_test)[:,1]
        
        # Dividing the screen into 2 parts to display the graphs -
        model_expander_col1, model_expander_col2 = model_expander.beta_columns((1,1))
        
        # Creating confusion matrix -
        conf_matrix = confusion_matrix(y_test, y_pred_class)
        
        # Plot to show confusion matrix - 
        plt.figure(figsize=(9,9))
        plt.style.use("seaborn-poster")
        plt.matshow(conf_matrix, cmap = "coolwarm")
        plt.gca().xaxis.tick_bottom()
        plt.title(f"Confusion Matrix \n Accuracy - {round(accuracy_score(y_test, y_pred_class), 3)} \n F1-score - {round(f1_score(y_test, y_pred_class), 3)}")
        plt.ylabel("Actual Class")
        plt.xlabel("Predicted Class")
        for(i, j), corr_value in np.ndenumerate(conf_matrix) :
            plt.text(j, i, corr_value, ha = "center", va = "center", fontsize = 20)
        model_expander_col1.pyplot()
        mpl.rcParams.update(mpl.rcParamsDefault)
        
        # Making AUC -
        fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        model_expander_col2.write(' ')
        model_expander_col2.write(' ')
        # Plotting the AUC
        plt.figure(figsize=(10,10))
        plt.style.use("bmh")
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        model_expander_col2.pyplot()
        mpl.rcParams.update(mpl.rcParamsDefault)
    
# ~~~~~~~~~~~~~~~~~ INITIAL HOME PAGE ~~~~~~~~~~~~~~~~~

# Setting the page layout -
st.set_page_config(layout = 'wide')
st.set_option('deprecation.showPyplotGlobalUse', False)

# Setting the title - 
image = Image.open('image2.png')

# Sometimes images are not in RGB mode, this can throw an error
# To handle the same - 
if image.mode != "RGB":
    image = image.convert('RGB')

# Setting the image width -

st.image(image, use_column_width=True)

# Sidebar navigation for users -
st.sidebar.header('Navigation tab -')
navigation_tab = st.sidebar.selectbox('Choose a tab', ('Home-Page', 'Analytics-Page', 'Customer Loyalty Calculator', 'Marketing Recommender'))


# Displaying pages according to the selection -

# Default page -
if navigation_tab == 'Home-Page':
    # Introduction about the project -
    st.write("""
         # ABC Grocery Mart
         
         This app is designed to assist ABC Grocery Mart. It contains 3 sections - 
         * **Analytics-Page:** An integrated Tableau dashboard to give an overview of the business to the stakeholders.
         * **Customer Stores:** The users can get the loyalty score for a customer using a machine learning model.
         * **Recommendation Engine:** Employees can use this to decide whether or not to target a particular customer for the marketing campaign.
         
         """)
    st.write(' ')
    st.info('Please scroll through different sections using the navigation tab on the left')
    
    
    st.write(' ')
    
# Analytics Page -
elif navigation_tab == 'Analytics-Page':
    analytics_Page()

# Customer loyalty page -
elif navigation_tab == 'Customer Loyalty Calculator':
    customer_loyalty_calculator()

# Marketing recommendation page -
elif navigation_tab == 'Marketing Recommender':
    marketing_recommender()
    