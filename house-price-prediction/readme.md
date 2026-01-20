# Housing Price Prediction Model

A machine learning project that predicts median house values using the California Housing dataset. The model uses Random Forest Regression with a complete preprocessing pipeline for handling numerical and categorical features.

## Features

- **Stratified Train-Test Split**: Ensures balanced income distribution across training and test sets
- **Robust Preprocessing Pipeline**: 
  - Handles missing values with median imputation
  - Standardizes numerical features
  - One-hot encodes categorical features
- **Random Forest Regressor**: Ensemble model for accurate predictions
- **Model Persistence**: Saves trained model and pipeline for reuse
- **Inference Mode**: Load saved model to make predictions on new data

## Project Structure

```
house-price-prediction/
│
├── housing.csv               # Original dataset
└── house-price-prediction.py
```

## Requirements

```bash
pandas
numpy
scikit-learn
joblib
```

Install dependencies:
```bash
pip install pandas numpy scikit-learn joblib
```

## Usage

### Training Mode (First Run)

When you run the script for the first time, it will:

1. Load the housing dataset
2. Create stratified train-test split based on income categories
3. Build and fit the preprocessing pipeline
4. Train a Random Forest model
5. Save the model and pipeline as `.pkl` files
6. Export test data to `input.csv`

```bash
python house-price-prediction2.py
```

Output:
```
Model is trained. Congrats!
```

### Inference Mode (Subsequent Runs)

Once the model is trained, the script automatically switches to inference mode:

1. Loads the saved model and pipeline
2. Reads test data from `input.csv`
3. Makes predictions
4. Saves results to `output.csv`

```bash
python house-price-prediction2.py
```

Output:
```
Inference Completed
Results saves to output.csv
```

## How It Works

### Data Preprocessing

**Income Stratification**: The dataset is split into 5 income categories to ensure representative sampling:
- Category 1: $0 - $15,000
- Category 2: $15,000 - $30,000
- Category 3: $30,000 - $45,000
- Category 4: $45,000 - $60,000
- Category 5: $60,000+

**Numerical Features Pipeline**:
- Median imputation for missing values
- Standard scaling (zero mean, unit variance)

**Categorical Features Pipeline**:
- One-hot encoding for `ocean_proximity`
- Handles unknown categories gracefully

### Model

**Random Forest Regressor** is used because it:
- Handles non-linear relationships well
- Reduces overfitting through ensemble averaging
- Provides robust predictions without extensive tuning

## Dataset

The California Housing dataset includes:

**Features**:
- `longitude`, `latitude`: Geographic coordinates
- `housing_median_age`: Median age of houses in block
- `total_rooms`, `total_bedrooms`: Total rooms/bedrooms in block
- `population`: Block population
- `households`: Number of households
- `median_income`: Median income (in tens of thousands)
- `ocean_proximity`: Categorical proximity to ocean

**Target**:
- `median_house_value`: Median house value in block

## Model Performance

To evaluate model performance, you can add cross-validation:

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, data_prepared, data_labels, 
                        scoring='neg_root_mean_squared_error', cv=10)
rmse_scores = -scores
print(f"RMSE: ${rmse_scores.mean():,.2f} (+/- ${rmse_scores.std():,.2f})")
```

## Customization

### Change Model Parameters

```python
model = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_depth=20,          # Maximum tree depth
    min_samples_split=10,  # Minimum samples to split
    random_state=42
)
```

### Try Different Models

```python
from sklearn.ensemble import GradientBoostingRegressor

model = GradientBoostingRegressor(random_state=42)
```

## File Descriptions

- **model.pkl**: Serialized Random Forest model
- **pipeline.pkl**: Serialized preprocessing pipeline (imputation, scaling, encoding)
- **input.csv**: Test set data (20% of original dataset)
- **output.csv**: Test set with predicted `median_house_value` column

## Troubleshooting

**Error: "could not convert string to float"**
- Ensure categorical features are properly encoded through the pipeline

**Error: "None of [...] are in the [index]"**
- Don't overwrite the original `data` variable before creating test split

**Model not retraining**
- Delete `model.pkl` to force retraining

## Future Improvements

- [ ] Add hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
- [ ] Feature engineering (rooms per household, bedrooms ratio, etc.)
- [ ] Compare multiple models (XGBoost, LightGBM, Neural Networks)
- [ ] Add model evaluation metrics on test set
- [ ] Create visualization of predictions vs actual values
- [ ] Add logging for better debugging
- [ ] Implement command-line arguments for flexibility

## License

This project is for educational purposes.

## Author

Created as a machine learning practice project using scikit-learn.