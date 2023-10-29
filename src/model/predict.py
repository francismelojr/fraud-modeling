# pipeline
import pickle
import pandas as pd

input_file = "src/model/model_lgbm.bin"

with open(input_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)

data = pd.read_csv('data/data.csv')

X = data.drop(columns=['fraude'])

print(pipeline.predict_proba(X))
