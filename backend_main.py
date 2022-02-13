# 1. Library imports
import uvicorn
from fastapi import FastAPI
import pandas as pd
from downcast import reduce
import shap
import pickle
import json
import gc
gc.enable()
from Credit_Scorer_Class import Credit_Scorer
from HomeCredit_Data import HomeCredit_Data


# 1. Create API app
app = FastAPI()

# 2. Import Trained LiIghtGBM Classifier
scorer = Credit_Scorer()

# 3. Set Index route, opens automatically on http://127.0.0.1:8000
@app.get("/")
def index():
    return {"Welcome to Home Credit Simulator"}


# 4.Create prediction endpoint
@app.post("/predict")  # , response_model = HomeCredit_Data)
async def predict(data: HomeCredit_Data):

    # Extract data in correct order
    customer_dict_data = data.dict()
    print("customer_dict", customer_dict_data)
    customer_df_data = pd.DataFrame(
        customer_dict_data.values(), index=customer_dict_data.keys()
    ).T
    customer_df_data = reduce(customer_df_data)

    # Predictions
    customer_prediction = scorer.predict(customer_df_data)
    print("customer_prediction", customer_prediction)

    proba_score = float(customer_prediction["probability"])
    credit_class = str(customer_prediction["credit_class"])
    client_cluster = int(customer_prediction["client_cluster"])
    transformed_data = customer_prediction["transformed_data"]
    
    # Calculate shap local values 
    explainer = pickle.load(open("./model_classifier/shap_explainer.pkl", "rb"))
    base_value = explainer.expected_value
    shap_local_values = explainer.shap_values(pd.DataFrame.from_dict(transformed_data))
    
    # Convert numpy arrays to JSON format
    base_value = json.dumps(base_value.tolist())
    shap_local_values = json.dumps(shap_local_values[1].tolist())
    
    # Return response back to client & clean memory
    del customer_df_data
    gc.collect()
    gc.collect()
    
    return {
        "probabilty": proba_score,
        "credit_class": credit_class,
        "client_cluster": client_cluster,
        "transformed_data": transformed_data,
        "shap_base_value": base_value,
        "shap_local_values": shap_local_values
    }



# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
# if __name__ == "__backend_main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000, debug=True)