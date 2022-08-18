
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~ GROCERY STORE APPLICATION ~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# ~~~~~~~~~~~~~~~~~~~~ Importing the packages ~~~~~~~~~~~~~~~~~~~~

import streamlit as st
from pages.common_functions.app_functions import title_image

# ~~~~~~~~~~~~~~~~~~~~ Grocery Store Application ~~~~~~~~~~~~~~~~~~~~

# Getting the title image - 
title_image()

# Introduction about the project -

st.write("""
     # Grocery Store Application
     
     This application is designed to assist Grocery Sotre Mart. It contains teh following sections -  
     * **Analytics-Page:** An integrated Tableau dashboard to give an overview of the business to the stakeholders.  
     * **Assessing Campaign Performance Using AB Testing:** In this project we apply Chi-Square Test For Independence (a Hypothesis Test) to assess the performance of two types of mailers that were sent out to promote a new service!  
     * **Predicting Customer Loyalty Scores:** ABC grocery retailer hired a market research consultancy to append market level customer loyalty information to the database.  However, only around 50% of the client's customer base could be tagged, thus the other half did not have this information present.  Let's use ML to solve this!
     * **Enhancing Customer Targeting Accuracy:** ABC Grocery sent out mailers in a marketing campaign for their new delivery club. Based upon the results of the last campaign and the customer data available, we will look to understand the probability of customers signing up for the delivery club. This would allow the client to mail a more targeted selection of customers, lowering costs, and improving ROI.
     * **Casual Impact Analysis:** In this project we use Causal Impact Analysis to analyse & understand the sales uplift of customers that joined the new "Delivery Club" campaign.
     * **Principal Component Analysis:** ABC grocery wants to know the potential buyers for the new Ed Sheeran's album. In this project we use Principal Component Analysis (PCA) to compress 100 unlabelled, sparse features into a more manageable number for classiying buyers of Ed Sheeran's latest album.
     * **Customer Segmentation:** In this project we use k-means clustering to segment up the customer base in order to increase business understanding, and to enhance the relevancy of targeted messaging & customer communications.
     * **Association Rule Learning:** In this project we use Association Rule Learning to analyse the transactional relationships & dependencies between products in the alcohol section of ABC retail grocery store.
     """)
st.write(' ')

# Getting attention to the navigation tab -
st.info('Please scroll through different sections using the navigation tab on the left.')

st.write(' ')

