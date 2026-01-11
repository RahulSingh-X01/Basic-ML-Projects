from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Features: [weight, color (0 = red, 1 = orange)]
features = [
    # Apples (red, lighter)
    [130, 0], [135, 0], [140, 0], [145, 0], [150, 0],
    
    # Oranges (orange, medium)
    [170, 1], [175, 1], [180, 1], [185, 1], [190, 1],
    
    # Bananas (yellow, lighter)
    [120, 2], [125, 2], [130, 2], [135, 2], [140, 2],
    
    # Kiwis (green, small)
    [70, 3], [75, 3], [80, 3], [85, 3], [90, 3],
]

labels = (["apple"] * 5 + ["orange"] * 5 + ["banana"] * 5 + ["kiwi"] * 5)

clf = DecisionTreeClassifier()

clf = clf.fit(features, labels)

print("Predictions:")
print(f"The fruit is : {clf.predict([[140, 0]])[0]}")
print(f"The fruit is : {clf.predict([[180, 1]])[0]}")
print(f"The fruit is : {clf.predict([[130, 2]])[0]}")
print(f"The fruit is : {clf.predict([[80, 3]])[0]}") 

plt.figure(figsize=(11,6))

plot_tree(clf, feature_names=["weight", "color"], class_names=["apple", "orange", "banana", "kiwi"], filled=True, fontsize=12)

plt.tight_layout()
plt.show()
