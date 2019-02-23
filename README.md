#Importing neccesary models

import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Importing the excel file "Iris.xls"

df = pd.read_excel("Iris.xls")

#Summarize the dataset
print(df.shape)
print(df.head(20))
print(df.describe())
print(df.groupby("iris").size())

#Data visualization
#showing the contribution of each atrributes one by one

df.plot(kind = "box", sharex = False, sharey = False, layout = (2,2), subplots = True)
plt.show()
df.hist()
plt.show()

#showing the contribution of attributes with each others

scatter_matrix(df)
plt.show()

#Data preparation

df1 = df.values
X = df1[:,0:4]
Y = df1[:,4]

#Spliting data into two, 80% for training and 20% for testing the dataset

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 7)

#Build models

Models = []
Models.append(("LR", LogisticRegression()))
Models.append(("LDA", LinearDiscriminantAnalysis()))
Models.append(("KNN", KNeighborsClassifier()))
Models.append(("BA", GaussianNB()))
Models.append(("DT", DecisionTreeClassifier()))
Models.append(("SVM", SVC()))

Results = []
Names = []
for name, model in Models:
    kfold = KFold(n_splits = 10, random_state = 7)
    cross_val = cross_val_score(model, X_train,Y_train, cv = kfold, scoring = 'accuracy')
    Results.append(cross_val)
    Names.append(name)
    print("{}, {}, {}".format(name, cross_val.mean(), cross_val.std()))*
    
#Models accuracy evaluation

fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(Results)
ax.set_xticklabels(Names)
plt.show()

#Make predicitons

SVM = SVC()
SVC.fit(X_train, Y_train)
predictions = SVM.predict(X_test)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
