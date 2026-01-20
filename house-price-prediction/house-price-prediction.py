import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attributes, cat_attributes):
    num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy='median')),
    ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown='ignore')),
    ])

    full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    
    data = pd.read_csv(r"house-price-prediction\housing.csv")
    
    data['income_cat'] = pd.cut(data['median_income'], bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf], labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_index, test_index in split.split(data, data['income_cat']):
        train_data = data.loc[train_index].drop('income_cat', axis=1)
        test_data = data.loc[test_index].drop('income_cat', axis=1)
        test_data.to_csv("input.csv", index=False)
        
    data_labels = train_data['median_house_value'].copy()
    data_features = train_data.drop('median_house_value', axis=1)

    num_attributes = data_features.drop('ocean_proximity', axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]
    
    pipeline = build_pipeline(num_attributes, cat_attributes)
    data_prepared = pipeline.fit_transform(data_features)
    
    model = RandomForestRegressor(random_state=42)
    
    model.fit(data_prepared, data_labels)
    
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    
    print("Model is trained. Congrats!")


else:
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    input_data = pd.read_csv("input.csv")
    transformed_input = pipeline.transform(input_data)
    
    predictions = model.predict(transformed_input)
    
    input_data['median_house_value'] = predictions
    
    input_data.to_csv("output.csv", index=False)

    print("Inference Completed\nResults saves to output.csv")
    
