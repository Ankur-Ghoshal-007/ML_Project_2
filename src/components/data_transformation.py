import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle

class DataTransformation:
    def __init__(self):
        pass

    def get_preprocessor(self):
        numerical_cols = ["Age","Tenure","Usage Frequency","Support Calls","Payment Delay","Total Spend","Last Interaction"]

        categorical_cols = ["Gender","Subscription Type","Contract Length"]
        
        num_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
        
        cat_pipeline = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

        preprocessor = ColumnTransformer(transformers=[('num', num_pipeline, numerical_cols),('cat', cat_pipeline, categorical_cols)])

        return preprocessor
    
    def initiate_data_transformation(self,df):

        numerical_cols = [
            "Age",
            "Tenure",
            "Usage Frequency",
            "Support Calls",
            "Payment Delay",
            "Total Spend",
            "Last Interaction"
        ]

        categorical_cols = [
            "Gender",
            "Subscription Type",
            "Contract Length"
        ]

        df['Total Spend'] = pd.to_numeric(df['Total Spend'], errors = 'coerce')
        # df.fillna(df.median(), inplace = True)

        df[numerical_cols] = df[numerical_cols].fillna(
        df[numerical_cols].median()
    )

        df[categorical_cols] = df[categorical_cols].fillna("Unknown")

        # df = df.drop('CustomerId', axis = 1)

        # df['Churn'] = df['Churn'].map({'yes':1, 'No':0})

        X = df.drop('Churn', axis = 1)
        Y = df['Churn']

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        preprocessor = self.get_preprocessor()

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)

        # Save preprocessor
        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/preprocessor.pkl", "wb") as f:
            pickle.dump(preprocessor, f)

        return (
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test
        )

