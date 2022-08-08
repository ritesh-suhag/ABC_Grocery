
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