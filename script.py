import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Loader datasættet fra UCI i form at et URL.
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
# Tildeler navne til de komma separeret kolonne værdier.
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
# Bruger pandas read_csv funktion til at referere til datasættet og fortælle hvilke navne den skal bruge. Gemmer det derefter i en variabel.
dataset = pd.read_csv(url, names=names)

# Fortæller hvor mange rows og columns der er.
print(dataset.shape)

# Viser et udkast af datasættet med det antal rows som du angiver i head funktionen.
print(dataset.head(10))

# Viser basically et box plot i tekstform
print(dataset.describe())

# Gruppere dataen med f.eks 'class'. Så viser den hvor mange værdier der tilhører den værdi. (hvor mange blomster der tilhører den klasse)
print(dataset.groupby('class').size())

# Viser et boxplot over sepal-length, sepal-width, petal-length og petal-width.
# dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
# plt.show()

# Histogram over sepal-length, sepal-width, petal-length og petal-width.
# dataset.hist()
# plt.show()

# Laver et array med værdierne i datasættet. Ligner utrolig meget datasættet i CSV form, bare uden kommaer.
array = dataset.values

# Laver alle værdier i alle kolonner udover 'class' om til en X variabel
X = array[:, 0:4]

# Laver alle værdier i class kolonnen om til en Y variabel
Y = array[:, 4]

# Bare en variabel som vi bruger til at angive det samme seed for hver algoritme, så alle modeller har det samme udgangspunkt.
seed = 12

# Splitter dataen op i et train test split som giver os mulighed for at teste en algoritme på en del af dataen.
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.50, random_state=seed, shuffle=True)

scoring = 'accuracy'

# Laver en tom liste ved navn models. Denne lister fylder vi med forskellige algoritmer.
models = []
# Tilføjer algoritmerne til vores liste.
models.append(('LR', LogisticRegression(max_iter=4000)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Laver en tom liste som kommer til at indeholde vores resultater
results = []

# Laver en tom liste som indeholder navnene på vores modeller
names = []

for name, model in models:
    # Bruger k-fold cross-validation til at estimere models 'skill'
    kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
    # Gemmer cross validation scoren af vores model i variablen cv_results
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    # Tilføjer resultaterne til vores results liste
    results.append(cv_results)
    # Tilføjer navnet på modellen til listen med navne
    names.append(name)
    # Skriver navn på modellen, medianen og standardafvigelsen ud til consolen
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# Sammenligner vores algoritmer i form af box plot
fig = plt.figure()
fig.suptitle('Algoritme sammenligning')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#SVC
'''svm = SVC()
svm.fit(X_train, Y_train)
predictions = svm.predict(X_test)'''

#LDA
'''ldam = LinearDiscriminantAnalysis()
ldam.fit(X_train, Y_train)
predictions = ldam.predict(X_test)'''

#LR
'''lrm = LogisticRegression()
lrm.fit(X_train, Y_train)
predictions = lrm.predict(X_test)'''

#KNN
'''knnm = KNeighborsClassifier()
knnm.fit(X_train, Y_train)
predictions = knnm.predict(X_test)'''

#NB
'''nbm = GaussianNB()
nbm.fit(X_train, Y_train)
predictions = nbm.predict(X_test)'''

#DTC
'''dtcm = DecisionTreeClassifier()
dtcm.fit(X_train, Y_train)
predictions = dtcm.predict(X_test)'''

# Viser modellens nøjagtighedsprocent
#print(accuracy_score(Y_test, predictions))

# Viser et matrix som viser alle vores true positives, false positives, false negatives og true positives. Altså hvor mange den gættede rigtigt og forkert, bare delt op
#print(confusion_matrix(Y_test, predictions))

# Viser et mere in depth look på accuracy
#print(classification_report(Y_test, predictions))
