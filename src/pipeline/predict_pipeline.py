import os
import sys
import numpy as np
import pandas as pd

from src.exceptions import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            # paths of saved objects
            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            # loading saved model and preprocessor
            model = load_object(model_path)

            preprocessor = load_object(preprocessor_path)

            # transforming input data
            data_scaled = preprocessor.transform(features)

            # prediction
            prediction = model.predict(data_scaled)

            return prediction[0]

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        Gender: str,
        Married: str,
        Dependents: str,
        Education: str,
        Self_Employed: str,
        ApplicantIncome: float,
        CoapplicantIncome: float,
        LoanAmount: float,
        Loan_Amount_Term: float,
        Credit_History: float,
        Property_Area: str
    ):

        self.Gender = Gender
        self.Married = Married
        self.Dependents = Dependents
        self.Education = Education
        self.Self_Employed = Self_Employed
        self.ApplicantIncome = ApplicantIncome
        self.CoapplicantIncome = CoapplicantIncome
        self.LoanAmount = LoanAmount
        self.Loan_Amount_Term = Loan_Amount_Term
        self.Credit_History = Credit_History
        self.Property_Area = Property_Area

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "ApplicantIncome": [self.ApplicantIncome],
                "CoapplicantIncome": [self.CoapplicantIncome],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History": [self.Credit_History],
                "Property_Area": [self.Property_Area]
            }

            # converting dictionary to dataframe
            df = pd.DataFrame(custom_data_input_dict)

            # feature engineering

            df["TotalIncome"] = (
                df["ApplicantIncome"] +
                df["CoapplicantIncome"]
            )

            df["LoanIncomeRatio"] = (
                df["LoanAmount"] /
                (df["TotalIncome"] + 1)
            )

            df["TotalIncome_Log"] = np.log1p(
                df["TotalIncome"]
            )

            df["LoanAmount_Log"] = np.log1p(
                df["LoanAmount"]
            )

            return df

        except Exception as e:
            raise CustomException(e, sys)