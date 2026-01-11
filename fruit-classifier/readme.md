# Fruit Classifier

A beginner-friendly machine learning project that demonstrates classification algorithms by identifying fruits based on their physical characteristics. This project includes two implementations using Decision Tree and Random Forest algorithms, allowing you to compare single-tree and ensemble learning approaches.

## 📋 Overview

This project showcases how machine learning can classify fruits using simple features like weight and color. By implementing both Decision Tree and Random Forest classifiers, you'll understand the difference between using a single decision-making model versus combining multiple models for improved accuracy.

## 🎯 What You'll Learn

- **Supervised Learning**: Training models with labeled data
- **Classification Algorithms**: Decision Trees and Random Forests
- **Feature Engineering**: Encoding categorical data (colors) as numerical values
- **Ensemble Methods**: How combining multiple trees improves predictions
- **Model Visualization**: Understanding decision-making processes through tree diagrams
- **scikit-learn Library**: Practical implementation of ML algorithms

## 📂 Project Structure

```
fruit-classifier/
├── README.md
├── requirements.txt
├── decision-tree/
│   ├── fruit_classifier_dt.py
│   └── README.md
└── random-forest/
    ├── fruit_classifier_rf.py
    └── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.x
- pip package manager

### Installation

1. Clone or download this project
2. Install required dependencies:

```bash
pip install scikit-learn matplotlib
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

### Running the Projects

**Decision Tree Classifier:**
```bash
cd decision-tree
python fruit_classifier_dt.py
```

**Random Forest Classifier:**
```bash
cd random-forest
python fruit_classifier_rf.py
```

## 📊 Projects Included

### 1. Decision Tree Classifier

**File**: `decision-tree/fruit_classifier_dt.py`

A single decision tree model that classifies four types of fruits: apples, oranges, bananas, and kiwis.

**Features:**
- Multi-class classification (4 fruit types)
- Dataset: 20 samples (5 per fruit)
- Features: weight (grams) and color (encoded as integers)
- Visualization: Complete tree structure showing decision paths

**Dataset:**
| Fruit  | Weight Range | Color Code | Samples |
|--------|--------------|------------|---------|
| Apple  | 130-150g     | 0 (Red)    | 5       |
| Orange | 170-190g     | 1 (Orange) | 5       |
| Banana | 120-140g     | 2 (Yellow) | 5       |
| Kiwi   | 70-90g       | 3 (Green)  | 5       |

**Output Example:**
```
Predictions:
The fruit is : apple
The fruit is : orange
The fruit is : banana
The fruit is : kiwi
```

**Key Concepts:**
- Single tree makes all decisions
- Splits data based on feature thresholds
- Easy to interpret and visualize
- Can overfit on small datasets

### 2. Random Forest Classifier

**File**: `random-forest/fruit_classifier_rf.py`

An ensemble model using 10 decision trees to classify apples and oranges with improved accuracy.

**Features:**
- Binary classification (2 fruit types)
- Dataset: 13 samples (7 apples, 6 oranges)
- Features: weight (grams) and color (encoded as integers)
- Visualization: Shows one tree from the forest of 10
- Uses majority voting for final prediction

**Dataset:**
| Fruit  | Weight Range | Color Code | Samples |
|--------|--------------|------------|---------|
| Apple  | 130-160g     | 0 (Red)    | 7       |
| Orange | 170-200g     | 1 (Orange) | 6       |

**Output Example:**
```
The fruit is : apple
The fruit is : orange
The fruit is : apple
```

**Key Concepts:**
- Ensemble of 10 trees working together
- Each tree trained on random data subset
- Final prediction by majority vote
- More robust and less prone to overfitting
- Better accuracy than single tree

## 🔄 Comparison: Decision Tree vs Random Forest

| Aspect | Decision Tree | Random Forest |
|--------|---------------|---------------|
| **Model Type** | Single tree | Ensemble of 10 trees |
| **Fruit Types** | 4 (apple, orange, banana, kiwi) | 2 (apple, orange) |
| **Dataset Size** | 20 samples | 13 samples |
| **Prediction Method** | One tree decides | Majority vote of 10 trees |
| **Overfitting Risk** | Higher | Lower |
| **Training Speed** | Faster | Slower |
| **Accuracy** | Good | Better |
| **Interpretability** | Very clear | More complex |
| **Best For** | Understanding basics | Production use |

## 🎓 Learning Path

1. **Start with Decision Tree**: Understand how a single tree makes decisions
2. **Study the Visualization**: See how the tree splits data at each node
3. **Move to Random Forest**: Learn how multiple trees work together
4. **Compare Outputs**: Notice how ensemble methods improve reliability
5. **Experiment**: Modify parameters and observe changes

## 🛠️ Customization Ideas

### For Both Projects:

**Add More Features:**
```python
# Include diameter, texture, etc.
features = [[weight, color, diameter, texture], ...]
```

**Increase Dataset Size:**
```python
# Add more samples for better training
features = [[130, 0], [132, 0], [134, 0], ...]  # More variations
```

**Test with Real Data:**
```python
# Try your own measurements
print(clf.predict([[165, 0]]))  # What fruit is this?
```

### Decision Tree Specific:

```python
# Control tree depth
clf = DecisionTreeClassifier(max_depth=3)

# Set minimum samples per split
clf = DecisionTreeClassifier(min_samples_split=3)
```

### Random Forest Specific:

```python
# Increase number of trees
clf = RandomForestClassifier(n_estimators=100)

# Check feature importance
importances = clf.feature_importances_
print(f"Weight importance: {importances[0]:.2f}")
print(f"Color importance: {importances[1]:.2f}")

# Visualize different trees
plot_tree(clf.estimators_[5], ...)  # Show 6th tree
```

## 📈 Expected Outputs

Both projects will:
1. Print predictions for test samples
2. Display a matplotlib window with tree visualization(s)
3. Show decision nodes and splitting criteria
4. Demonstrate classification confidence with color coding

## 🧪 Experiment Suggestions

1. **Modify Weights**: Change fruit weights and see how predictions change
2. **Add Noise**: Include outliers to test model robustness
3. **Compare Accuracy**: Test both models on same data
4. **Visualize All Trees**: In Random Forest, plot multiple trees to see variations
5. **Feature Engineering**: Add new features like shape or sweetness ratings

## 🐛 Troubleshooting

**Issue**: Import errors for sklearn
```bash
# Solution: Install scikit-learn
pip install scikit-learn
```

**Issue**: Matplotlib window doesn't show
```bash
# Solution: Install proper backend
pip install pyqt5
```

**Issue**: Trees look too complex
```python
# Solution: Limit tree depth
clf = DecisionTreeClassifier(max_depth=3)
```

## 📚 Additional Resources

- [scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [scikit-learn Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Understanding Decision Trees](https://en.wikipedia.org/wiki/Decision_tree_learning)
- [Ensemble Learning Explained](https://en.wikipedia.org/wiki/Ensemble_learning)

## 🎯 Key Takeaways

- **Decision Trees** are intuitive and easy to visualize but can overfit
- **Random Forests** combine multiple trees for better accuracy and robustness
- **Feature encoding** is essential for handling categorical data like colors
- **Visualization** helps understand how models make decisions
- **Ensemble methods** generally outperform single models in production

## 🤝 Contributing

Feel free to enhance this project by:
- Adding more fruit types
- Including additional features
- Implementing other classifiers (SVM, KNN)
- Adding train/test splits and evaluation metrics
- Creating interactive visualizations

## 📝 License

This is an educational project. Feel free to use and modify for learning purposes.

## 🙏 Acknowledgments

Built with scikit-learn and matplotlib. Perfect for anyone beginning their machine learning journey!

---

**Happy Learning! Start with Decision Tree, then explore Random Forest to see the power of ensemble methods! 🍎🍊🍌🥝**