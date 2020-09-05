# BigMart Sales Prediction Analysis - Regression

**Complete Video Tutorial:** https://youtu.be/CD58mco2XqA

# Dataset Information

The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.

Using this model, BigMart will try to understand the properties of products and stores which play a key role in increasing sales.


Variable | Description
----------|--------------
Item_Identifier | Unique product ID
Item_Weight | Weight of product
Item_Fat_Content | Whether the product is low fat or not
Item_Visibility | The % of total display area of all products in a    store allocated to the particular product
Item_Type | The category to which the product belongs
Item_MRP | Maximum Retail Price (list price) of the product
Outlet_Identifier | Unique store ID
Outlet_Establishment_Year | The year in which store was established
Outlet_Size | The size of the store in terms of ground area covered
Outlet_Location_Type | The type of city in which the store is located
Outlet_Type | Whether the outlet is just a grocery store or some sort of supermarket
Item_Outlet_Sales | Sales of the product in the particulat store. This is the outcome variable to be predicted.

**Download link:** https://www.kaggle.com/devashish0507/big-mart-sales-prediction

# Libraries

<li>pandas
<li>matplotlib
<li>seaborn
<li>scikit-learn

# Algorithms

<li>Linear Regression
<li>Ridge
<li>Lasso
<li>Decision Tree
<li>Random Forest
<li>Extra Tress
  
**Mean Squared Error:** 0.28

**Note:** The Error metric is in log values. To convert to original values, use exponentiation.
