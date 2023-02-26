# Mastering Machine Learning with PySpark - Loan Prediction

**Complete Video Tutorial:** https://youtu.be/2OAa7lX8dxo

# Project Information

The goal of this project is to demonstrate the use of PySpark and Machine Learning to predict loan approvals. The project will involve the following steps:

1. Data Collection: Collecting data related to loan approvals, including features such as income, credit score, loan amount, and loan status.

2. Data Preparation: Preprocessing and cleaning up the data to ensure it is ready for use in Machine Learning algorithms.

3. Feature Engineering: Selecting and engineering relevant features to improve the accuracy of the model.

4. Model Training: Using PySpark to train a Machine Learning model on the prepared data.

5. Model Evaluation: Testing the trained model on a validation set to evaluate its accuracy and adjust the parameters if necessary.

6. Loan Prediction: Using the trained model to predict whether a loan will be approved or not based on input features.

The project will require knowledge of PySpark, Machine Learning, and Python programming. The end result will be a Machine Learning model that can accurately predict loan approvals based on various features. This can be used by financial institutions or individuals to make more informed decisions regarding loans. Additionally, the project can be extended by exploring different Machine Learning algorithms and improving the accuracy of the model.

# Dataset Information

   Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.
   
   This is a standard supervised classification task.A classification problem where we have to predict whether a loan would be approved or not. Below is the dataset attributes with description.
   
Variable | Description
----------|--------------
Loan_ID | Unique Loan ID
Gender | Male/ Female
Married | Applicant married (Y/N)
Dependents | Number of dependents
Education | Applicant Education (Graduate/ Under Graduate)
Self_Employed | Self employed (Y/N)
ApplicantIncome | Applicant income
CoapplicantIncome | Coapplicant income
LoanAmount | Loan amount in thousands
Loan_Amount_Term | Term of loan in months
Credit_History | credit history meets guidelines
Property_Area | Urban/ Semi Urban/ Rural
Loan_Status | Loan approved (Y/N)

**Download link:** https://www.kaggle.com/altruistdelhite04/loan-prediction-problem-dataset

# Libraries

<li>pyspark

# Algorithms

<li>Logistic Regression
<li>Random Forest
  
**Best Model AUC:** 82.00