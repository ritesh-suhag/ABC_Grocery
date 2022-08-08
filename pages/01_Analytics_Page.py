
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ANALYTICS PAGE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~ Importing the packages ~~~~~~~~~~~~~~~~~~~~

import streamlit as st
import streamlit.components.v1 as components
from pages.common_functions.app_functions import title_image

# ~~~~~~~~~~~~~~~~~~~~ Analytics page ~~~~~~~~~~~~~~~~~~~~

# Getting the title image - 
title_image()

# Introduction of the page -
st.write("""
         ## Welcome to the Analytics Page!
         
         This page is designed to give an overview of the business to the stakeholders. 
         """)
         
# Description of the charts -
st.write("""
                     The Dashboard consists of 4 Graphs - 
                     * **Sales vs Time:** The line graph shows the variations of sales over the 6 months by department. We can select one of  multiple departments to analyze at the same time.
                     * **Gender:** The pie chart shows the percent of male and female shopping at the stores. This chart can also be used filter values in the dashboard.
                     * **Distance from Stores:** The bar graph shows the distribution of customers according to their distances from the respective store locations. The graph is colored according to the customer loyalty scores. (Based on research from a consultancy, loyalty measures the proportion of a customer's total grocery spend that is allocated to ABC Grocery vs. their competitors)
                     * **Total Sales State Map:** The map shows the sales distribution across the states in the US.
                     """)
st.write('---')

# ~~~~~~~~~~ Loading in the tableau dashboard -

st.write("## Tableau Dashboard")

# The HTML address of the dashboard (copied from tableau)
html_temp = """
            <div class='tableauPlaceholder' id='viz1643071209764' style='position: relative'><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='ABCGroceryDashboard&#47;Final' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1643071209764');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1000px';vizElement.style.height='827px';} else { vizElement.style.width='100%';vizElement.style.height='1427px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
            """
        
# Rendering in the dashboard -
components.html(html_temp, width=1130, height=900)
