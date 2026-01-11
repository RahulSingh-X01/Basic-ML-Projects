# Fruit Classifier Using Decision Trees

A simple machine learning project that demonstrates classification using Decision Trees to identify different fruits based on their weight and color.

## Overview

This project uses scikit-learn's Decision Tree algorithm to classify fruits into four categories: apples, oranges, bananas, and kiwis. The classifier learns patterns from training data containing weight and color features, then predicts the fruit type for new inputs.

## Features

- **Binary classification approach** using decision tree algorithm
- **Visual tree representation** showing the decision-making process
- **Simple feature set**: weight (grams) and color (encoded as integers)
- **Four fruit categories**: apple, orange, banana, and kiwi

## Dataset

The training dataset consists of 20 samples (5 per fruit type):

| Fruit  | Weight Range | Color Code | Color Name |
|--------|--------------|------------|------------|
| Apple  | 130-150g     | 0          | Red        |
| Orange | 170-190g     | 1          | Orange     |
| Banana | 120-140g     | 2          | Yellow     |
| Kiwi   | 70-90g       | 3          | Green      |

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
python fruit_classifier.py
```

The program will:
1. Train a decision tree classifier on the fruit dataset
2. Make predictions on test samples
3. Display a visualization of the decision tree structure

## Example Output

```
Predictions:
The fruit is : apple
The fruit is : orange
The fruit is : banana
The fruit is : kiwi
```

A matplotlib window will appear showing the decision tree with:
- Decision nodes (feature thresholds)
- Leaf nodes (final classifications)
- Sample counts and class distributions
- Color-coded confidence levels

## How It Works

1. **Feature Engineering**: Each fruit is represented by two features:
   - Weight in grams (continuous variable)
   - Color as an integer code (categorical variable)

2. **Training**: The DecisionTreeClassifier learns to split the data based on these features to create distinct categories

3. **Prediction**: New fruit samples are classified by traversing the decision tree based on their weight and color values

4. **Visualization**: The tree structure shows how decisions are made at each node to reach a final classification


## Learning Outcomes

This project demonstrates:
- Basic supervised learning concepts
- Decision tree classification
- Feature encoding for categorical data
- Model training and prediction workflow
- Visualization of ML models

## Limitations

- The model is trained on a small synthetic dataset
- Color encoding is simplified (real fruits have complex color patterns)
- Weight ranges may overlap in real-world scenarios
- No train/test split or validation (for educational simplicity)

## License

This is an educational project. Feel free to use and modify for learning purposes.