
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMMON APP FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~ Importing the packages ~~~~~~~~~~~~~~~~~~~~

import streamlit as st
from PIL import Image

def plot_image_121(image_name, image_folder, partition_list = [1,3,1]):
    # Setting the title - 
    image = Image.open(f'pages/Images/{image_folder}/{image_name}.png')

    # Sometimes images are not in RGB mode, this can throw an error
    # To handle the same - 
    if image.mode != "RGB":
        image = image.convert('RGB')
        
    # Setting the image width -
    col1, col2, col3 = st.columns(partition_list)
    col2.image(image)
    
def title_image():
    # Setting the page layout -
    st.set_page_config(layout = 'wide', page_title = "Grocery Store Application")
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Setting the title - 
    image = Image.open('pages/Images/grocery_store_title.png')

    # Sometimes images are not in RGB mode, this can throw an error
    # To handle the same - 
    if image.mode != "RGB":
        image = image.convert('RGB')

    # Setting the image width -
    st.image(image, use_column_width=True) 

input_check = [1,1,1,1,1,1]
def check_values(value_to_check, index_value, st_column, is_int = False, has_limits = False, upper_limit = 0, lower_limit = 0, input_check = input_check):
    
    try :
        if is_int:
            value_to_check = int(float(value_to_check))
        else:
            value_to_check = float(value_to_check)
        
        if has_limits:
            if value_to_check > upper_limit or value_to_check < lower_limit:
                input_check[index_value] = 0
                st_column.info(f'Please enter a number between {lower_limit} and {upper_limit}')
            else:
                input_check[index_value] = 1
        else:
            input_check[index_value] = 1
            
    except:
        input_check[index_value] = 0
        st_column.info('Please enter a number.')
    
    return input_check

# Function to get prediction -
def prediction_input_ui():
    
    # Setting a flag to ensure all correct inputs have been entered -
    input_check = [1,1,1,1,1,1]
    
    # ~~~~~~~~~~~~~ Getting in all the inputs and checking if they are as desired -
    
    # Dividing the window into 2 parts to get the input -
    input_col1, input_col2, input_col3 = st.columns((1,1,1))
    
    # Getting distance from store - 
    distance_from_store = input_col1.text_input('Distance from store (in miles). Mostly between 0 & 5', value = 1.3)
    input_check = check_values(distance_from_store, 0, input_col1)

    # No check required as it is a select box -
    gender_M = input_col1.selectbox('Gender', ['M', 'F'])
    
    # Getting credit score between 0 and 1
    credit_score = input_col2.text_input('Credit Score (Between 0 and 1)', value = 0.5)
    input_check = check_values(credit_score, 1, input_col2, has_limits = True, upper_limit=1, lower_limit=0)

    # Getting Total sales of the customer -
    total_sales = input_col2.text_input('Total Sales ($). Please enter numeric values', 100)
    input_check = check_values(total_sales, 2, input_col2)

    # Getting total items - 
    total_items = input_col3.text_input('Total items. Please enter numeric values', 10)
    input_check = check_values(total_items, 3, input_col3, is_int=True)

    # Getting the product area name -
    product_area_count = input_col3.text_input('Product Area Count (between 1 and 5)', 3)
    input_check = check_values(product_area_count, 4, input_col3,  is_int=True, has_limits = True, upper_limit=5, lower_limit=1)

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

    col1, col2, col3 = st.columns((1,3,1))
    
    if sum(input_check) != 6:
        col2.info(' Check all the inputs!')
        check_inputs = False
    else:
        check_inputs = True
    
    return check_inputs, distance_from_store, gender_M, credit_score, total_sales, total_items, product_area_count, transaction_count

