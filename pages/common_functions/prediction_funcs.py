#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 18:54:08 2022

@author: riteshsuhag
"""

## Deploying the Model -
"""
In order to understand how the results are fetched, see below.

#### Creating the API -

The API is created using FastAPI. FastAPI is a Python web framework that makes it easy for developers to build fast (high-performance), production-ready REST APIs. The purpose of deploying ML model as a FastAPI endpoint is to readily retrieve prediction results after parsing new data into the API. A few advantages of FastAPI are -

* Fast development
* Fewer bugs
* High and fast performance
* Validation of Input

FastAPI has built in capacity to use 'BaseModel' from 'pydantic' package which helps us control what type of input is sent into  the API. With the help of BaseModel, FastAPI automatically checks the input being sent into the API and returns error if wrong input is sent. This helps with security 
"""
import streamlit as st
