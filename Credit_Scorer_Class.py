import pickle
import warnings

warnings.filterwarnings("ignore")
import pandas as pd
import shap
import sklearn.neighbors._base
import sys

sys.modules["sklearn.neighbors.base"] = sklearn.neighbors._base


class Credit_Scorer:
    def __init__(
        self,
        model_path="model_classifier/model_package.pkl",
    ):

        model_package = pickle.load(open(model_path, "rb"))  # Load model package
        self.categorical_imputer = model_package[
            "categorical_imputer"
        ]  # Load Categorical Features Missing Data Imputer
        self.categorical_features = model_package[
            "categorical_features"
        ]  # Load Categorical Features List
        self.boolean_imputer = model_package[
            "boolean_imputer"
        ]  # Load Boolean Features Missing Data Imputer
        self.boolean_fatures = model_package[
            "boolean_fatures"
        ]  # Load Boolean Features List
        self.numerical_imputer = model_package[
            "numerical_imputer"
        ]  # Load Numerical Features Missing Data Imputer
        self.numerical_features = model_package[
            "numerical_features"
        ]  # Load Numerical Features List
        self.data_preprocessor = model_package[
            "preprocess_data"
        ]  # Load Data Preprocessor (OneHot + Scaler)
        self.drop_features = model_package[
            "drop_features"
        ]  # Load Features List to be dropped
        self.model = model_package[
            "model"
        ]  # Load LightGBM model to predict Credit Home Class
        self.threshold = model_package["threshold"]  # Load threshold
        self.features_after_preprocessor = model_package[
            "features_after_preprocessor"
        ]  # Feature List after processor step
        self.kmeans = model_package["kmeans_clustering"]  # Kmenas clustering

    def predict(self, customer_data):
        """
        Use LightGBM estimator, to calculate the probability of a customer refunding a credit.
        The output is the probability score and credit request classification (credit granted or refused).

        Parameters
        ----------
        X : Dictionary, containing data to predict on.

        Returns
        -------
        probabilty_score :
        credit_class :
        """

        # =================== Step 0 : Remove customer ID ================================================================================#

        print("step n°1 : Remove SK_ID_CRR and TARGET columns if present")
        if "SK_ID_CURR" in customer_data.columns.tolist():
            customer_data.drop(
                "SK_ID_CURR", axis=1, inplace=True
            )  # Remove 'SK_ID_CURR' because it not used in the model

        elif "TARGET" in customer_data.columns.tolist():
            customer_data.drop(
                "TARGET", axis=1, inplace=True
            )  # Remove 'TARGET' because it not used in the model
        else:
            pass

        print("step n°1 : Completed Successfully")

        # =================== Step 1 : Impute missing data ===============================================================================#

        # Impute Categorical Missing values
        print("step n°2 : missing data Imputation")
        print("  - step n°2.1 : categorical data Imputation")

        customer_data[self.categorical_features] = self.categorical_imputer.transform(
            customer_data[self.categorical_features]
        )
        print("  - step n°2.1 : Completed Successfully")

        # Impute Boolean Missing values
        print("  - step n°2.2 : boolean data Imputation")
        customer_data[self.boolean_fatures] = self.boolean_imputer.transform(
            customer_data[self.boolean_fatures]
        )
        print("  - step n°2.2 : Completed Successfully")

        # Impute Numerical Missing values
        print("  - step n°2.3 : numerical data Imputation")
        customer_data[self.numerical_features] = self.numerical_imputer.transform(
            customer_data[self.numerical_features]
        )
        print("  - step n°2.3 : Completed Successfully")
        print("step n°2 : Completed Successfully")

        # =================== Step 2 : Preprocess data (OneHotEncoding and RobustScaling) =================================================#

        print("step n°3 : transform data (OneHotEncoding & RobustScaling)")
        preprocessed_data = self.data_preprocessor.transform(customer_data)

        preprocessed_data = pd.DataFrame(
            preprocessed_data, columns=self.features_after_preprocessor
        )
        print("step n°3 : Completed Successfully")

        # ==================== Step 3 : Feature Selection =================================================================================#

        print("step n°4 : feature selection")
        preprocessed_data.drop(self.drop_features, axis=1, inplace=True)
        print("step n°4 : Completed Successfully")

        # Ensure all features are in numerical format
        print("step n°5 : ensure all feature are in numerical format")
        for column in preprocessed_data.columns.to_list():
            preprocessed_data[column] = pd.to_numeric(preprocessed_data[column])
        print("step n°5 : Completed Successfully")

        # ==================== Step 4 : Calculate probability score and credit class ======================================================#

        print("step n°6 : predict customer credit score")
        proba_score = float(self.model.predict_proba(preprocessed_data)[0][1])

        if proba_score > self.threshold:
            credit_class = "crédit refusé"

        else:
            credit_class = "crédit accordé"
        print("step n°6 : Completed Successfully")

        # ==================== Step 5 : Identify client cluster ===========================================================================#

        print("step n°5 : Find client cluster")
        client_cluster = int(self.kmeans.predict(preprocessed_data))
        print("step °5 : Completed Successfully")

        # ==================== Step 6 : Retunr all calculated variables ===================================================================#

        print(
            "step n°6 : return probability, credit_class, client_cluster, shap_tree_explainer, shape local values & transformed_data"
        )

        return {
            "probability": proba_score,
            "credit_class": credit_class,
            "client_cluster": client_cluster,
            "transformed_data": preprocessed_data.to_dict(),
        }