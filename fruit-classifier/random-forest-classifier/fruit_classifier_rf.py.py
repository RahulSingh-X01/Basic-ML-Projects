from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Features: [weight, color (0 = red, 1 = orange)]
features = [
    [130, 0],  # Apple
    [135, 0],  # Apple
    [140, 0],  # Apple
    [145, 0],  # Apple
    [150, 0],  # Apple
    [155, 0],  # Apple
    [160, 0],  # Apple
    
    [170, 1],  # Orange
    [175, 1],  # Orange
    [180, 1],  # Orange
    [185, 1],  # Orange
    [190, 1],  # Orange
    [200, 1],  # Orange
]

labels = ["apple", "apple", "apple", "apple", "apple", "apple", "apple",
         "orange", "orange", "orange", "orange", "orange", "orange"]

# Changed to RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10, random_state=42)

clf = clf.fit(features, labels)

print(f"The fruit is : {clf.predict([[130, 0]])[0]}") 
print(f"The fruit is : {clf.predict([[200, 1]])[0]}") 
print(f"The fruit is : {clf.predict([[160, 0]])[0]}")  

plt.figure(figsize=(11, 6))

plot_tree(clf.estimators_[0], feature_names=["weight", "color"], class_names=["apple", "orange"], filled=True, fontsize=20)

plt.tight_layout()
plt.show()