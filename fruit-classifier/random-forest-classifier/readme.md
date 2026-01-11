# Random Forest Fruit Classifier

A machine learning project demonstrating ensemble learning with Random Forest to classify fruits (apples and oranges) based on weight and color features.

## Overview

This project uses scikit-learn's Random Forest algorithm to classify fruits into two categories: apples and oranges. Unlike a single decision tree, Random Forest creates multiple trees and combines their predictions for more robust and accurate classification.

## What is Random Forest?

Random Forest is an **ensemble learning method** that:
- Creates multiple decision trees during training (10 trees in this implementation)
- Each tree makes its own prediction
- The final prediction is determined by majority voting across all trees
- Provides better accuracy and reduces overfitting compared to single decision trees

## Features

- **Ensemble classification** using 10 decision trees
- **Visual representation** of one tree from the forest
- **Simple feature set**: weight (grams) and color (encoded as integers)
- **Binary classification**: apple vs orange

## Dataset

The training dataset consists of 13 samples:

| Fruit  | Weight Range | Color Code | Color Name | Samples |
|--------|--------------|------------|------------|---------|
| Apple  | 130-160g     | 0          | Red        | 7       |
| Orange | 170-200g     | 1          | Orange     | 6       |

### Feature Details
- **Weight**: Measured in grams, ranges from 130g to 200g
- **Color**: Binary encoding (0 = red for apples, 1 = orange for oranges)

## Requirements

```
scikit-learn
matplotlib
```

Install dependencies using:
```bash
pip install scikit-learn matplotlib
```

## Usage

Run the script to train the model and see predictions:

```bash
python random_forest_classifier.py
```

The program will:
1. Train a Random Forest with 10 decision trees
2. Make predictions on test samples
3. Display a visualization of the first tree in the forest

## Example Output

```
The fruit is : apple
The fruit is : orange
The fruit is : apple
```

A matplotlib window will display the structure of the first decision tree from the forest, showing:
- Decision nodes with splitting criteria
- Leaf nodes with final classifications
- Sample distributions at each node
- Color-coded predictions

## How It Works

### Training Process
1. **Data Preparation**: Features (weight, color) are paired with labels (apple, orange)
2. **Forest Creation**: 10 different decision trees are created, each with slight variations
3. **Random Sampling**: Each tree is trained on a random subset of the data (bootstrap sampling)
4. **Feature Randomness**: Each split considers a random subset of features

### Prediction Process
1. New fruit features are passed through all 10 trees
2. Each tree makes an independent prediction
3. Final prediction is determined by majority vote
4. Example: If 7 trees say "apple" and 3 say "orange", the result is "apple"

### Key Parameters
- `n_estimators=10`: Number of trees in the forest
- `random_state=42`: Ensures reproducible results

## Advantages of Random Forest

- **Higher Accuracy**: Ensemble of trees typically performs better than a single tree
- **Reduced Overfitting**: Averaging multiple trees reduces variance
- **Robustness**: Less sensitive to noisy data or outliers
- **Feature Importance**: Can measure which features are most important for classification

## Customization Ideas

You can enhance this project by:

```python
# Increase number of trees
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Add more features
features = [[weight, color, diameter], ...]

# Include more fruit types
labels = ["apple", "orange", "banana", "grape", ...]

# Adjust tree depth
clf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)

# Get feature importance
importances = clf.feature_importances_
print(f"Weight importance: {importances[0]}")
print(f"Color importance: {importances[1]}")
```

## Visualization Notes

The code visualizes only the **first tree** (index 0) from the forest of 10 trees. Each tree in the forest may have a slightly different structure due to:
- Random bootstrap sampling of training data
- Random feature selection at each split

To visualize other trees from the forest:
```python
# Visualize the 5th tree (index 4)
plot_tree(clf.estimators_[4], ...)
```

## Learning Outcomes

This project demonstrates:
- **Ensemble learning** concepts
- **Random Forest classification** algorithm
- **Bootstrap aggregating (bagging)** technique
- Difference between single decision tree and forest of trees
- Model training, prediction, and visualization


## Limitations

- Small synthetic dataset (13 samples)
- Binary classification only (2 fruit types)
- Simplified color encoding
- No train/test split (educational demonstration)
- Overlapping weight ranges could cause ambiguity in real scenarios


## License

This is an educational project demonstrating Random Forest classification. Feel free to use and modify for learning purposes.