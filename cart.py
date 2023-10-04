import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from scipy import stats

df = pd.read_csv("data_banknote_authentication.txt", header=None,
                    names=['variance', 'skewness', 'curtosis', 'entropy', 'class'])

X = df.iloc[:, 0:4]
y = df.iloc[:, 4]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=1)

# gini index min
# clf = DecisionTreeClassifier(max_depth=5)

# gini index random
clf = DecisionTreeClassifier(splitter='random', max_depth=5)

# entropy min
# clf = DecisionTreeClassifier(criterion='entropy', max_depth=5)

# entropy random
# clf = DecisionTreeClassifier(criterion='entropy', splitter='random', max_depth=5)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# figure fontsize 8
plt.figure(figsize=(20, 12))
tree.plot_tree(clf, feature_names=['variance', 'skewness', 'curtosis', 'entropy'], class_names=['0', '1'], fontsize=8)
plt.show()




