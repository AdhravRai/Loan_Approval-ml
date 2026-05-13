import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import GridSearchCV

from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self,X_train,y_train,X_test,y_test,models,params ):

        try:
            model_report = {}
            trained_models = {}

            for model_name, model in models.items():

                logging.info(f"Training {model_name}")

                para = params[model_name]

                gs = GridSearchCV(
                    estimator=model,
                    param_grid=para,
                    cv=5,
                    scoring="f1",
                    n_jobs=-1,
                    verbose=1
                )

                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_

                best_model.fit(X_train, y_train)

                y_pred = best_model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)

                f1 = f1_score(y_test, y_pred)

                logging.info(f"{model_name} Accuracy: {accuracy}")

                logging.info(f"{model_name} F1 Score: {f1}")

                logging.info(
                    f"\nClassification Report for {model_name}:\n"
                    f"{classification_report(y_test,y_pred)}"
                )
                cm = confusion_matrix(y_test,y_pred)

                logging.info(f"Confusion Matrix:\n{cm}")

                model_report[model_name] = f1

                trained_models[model_name] = best_model

            return model_report, trained_models

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(
        self,
        train_array,
        test_array
    ):

        try:

            logging.info(
                "Splitting training and testing input data"
            )

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            models = {

                "Logistic Regression": LogisticRegression(random_state=42,max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Random Forest": RandomForestClassifier(random_state=42),
                "KNN": KNeighborsClassifier()

               
            }

            params = {

                "Logistic Regression": {
                     "C": [0.01, 0.1, 1, 10],
                    "solver": ["liblinear"],
                    "class_weight": [
                        None,
                        "balanced"
                    ]
                },

                "Decision Tree": {
                    "criterion": [
                        "gini",
                        "entropy"
                    ],
                    "max_depth": [3,5,7,10,None],

                    "min_samples_split": [2,5,10]
                },

                "Random Forest": {

                    "n_estimators": [50,100,200],

                    "max_depth": [5,10,None],

                    "min_samples_split": [2,5]
                },

                "KNN": {

                    "n_neighbors": [3,5,7,9 ],

                    "weights": [
                        "uniform",
                        "distance"
                    ]
                }
            }

            model_report, trained_models = self.evaluate_model(

                X_train=X_train,
                y_train=y_train,

                X_test=X_test,
                y_test=y_test,

                models=models,
                params=params
            )

            best_model_score = max(
                model_report.values()
            )

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model = trained_models[
                best_model_name
            ]

            logging.info(
                f"Best Model Found: {best_model_name}"
            )

            logging.info(
                f"Best F1 Score: {best_model_score}"
            )
            


            # minimum threshold check

            if best_model_score < 0.60:
                raise CustomException(
                    "No best model found with good score",
                    sys
                )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            accuracy = accuracy_score(
                y_test,
                predicted
            )

            f1 = f1_score(
                y_test,
                predicted
            )

            logging.info(
                f"Final Accuracy: {accuracy}"
            )

            logging.info(
                f"Final F1 Score: {f1}"
            )

            return {
                "model_name": best_model_name,
                "accuracy": accuracy,
                "f1_score": f1
            }

        except Exception as e:
            raise CustomException(e, sys)