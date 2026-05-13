import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "ApplicantIncome",
                "CoapplicantIncome",
                "LoanAmount",
                "Loan_Amount_Term",
                "TotalIncome",
                "LoanIncomeRatio",
                "TotalIncome_Log",
                "LoanAmount_Log"
            ]
            categorical_columns = [
                "Gender",
                "Married",
                "Dependents",
                "Education",
                "Self_Employed",
                "Property_Area",
                "Credit_History"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]
            )
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            logging.info("Preprocessor object created")
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded")

            # dropping a column
            train_df.drop(columns=["Loan_ID"], inplace=True)
            test_df.drop(columns=["Loan_ID"], inplace=True)

            train_df["Loan_Status"] = train_df["Loan_Status"].map({"N": 0,"Y": 1})
            test_df["Loan_Status"] = test_df["Loan_Status"].map({ "N": 0, "Y": 1 })
            # doing feature eng
            for df in [train_df, test_df]:

                df["TotalIncome"] = (df["ApplicantIncome"] + df["CoapplicantIncome"])

                df["LoanIncomeRatio"] = (df["LoanAmount"] /(df["TotalIncome"] + 1))

                df["TotalIncome_Log"] = np.log1p(df['TotalIncome'])

                df["LoanAmount_Log"] = np.log1p(df['LoanAmount'])

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "Loan_Status"

            input_feature_train_df = train_df.drop(
                columns=[target_column_name],
                axis=1
            )
            target_feature_train_df = train_df[
                target_column_name
            ]

            input_feature_test_df = test_df.drop(
                columns=[target_column_name],
                axis=1
            )
            target_feature_test_df = test_df[
                target_column_name
            ]
            logging.info("Applying preprocessing on train and test data")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
                
            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            logging.info(f"Train array shape: {train_arr.shape}")
            logging.info(f"Test array shape: {test_arr.shape}")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor pickle file saved")
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)