
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

d = datasets.load_breast_cancer()
B = d.data
y = d.target

X_train, X_test, y_train, y_test = train_test_split(B, y, test_size=0.2, random_state=42)

clf =  DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print metrics.f1_score(y_test,pred)

scores = cross_val_score(clf, B, y, cv=5, scoring='f1')
print scores.mean()
# print scores
# end of program

