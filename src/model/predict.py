import os
import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException

path = os.path.dirname(os.path.abspath(__file__))

input_file = os.path.join(path, 'model_lgbm.bin')

with open(input_file, 'rb') as f_in:
    pipeline, best_threshold = pickle.load(f_in)

app = FastAPI()


@app.post('/predict')
async def predict(request: dict):

    X = pd.DataFrame([request])
    y_score = pipeline.predict_proba(X)[:, 1]
    is_fraud = y_score > best_threshold

    result = {'fraud_probability': float(y_score), 'fraud': bool(is_fraud)}

    return result


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)
