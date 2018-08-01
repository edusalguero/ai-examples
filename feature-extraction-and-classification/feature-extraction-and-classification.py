import pandas as pd
import numpy
import matplotlib.pyplot as plt
import time

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, mean_squared_error


# from sklearn.tests.test_multiclass import n_classes

# import pac2_classifier_comparison as cc


def loadAndPreprocess(filename):
    # ------------------------------------------------------------------------------------------------
    print("Activity 1a: ")

    dataLabels = [
        'age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal',
        'class'
    ]

    data = pd.read_csv(filename, sep=',', header=None, names=dataLabels, na_values=["?"])

    print("Original data")
    print(data.ix[:, 0:])

    n = len(data)
    print("Amount of instances: " + str(n))
    print("Analyzing class distribution")
    print(list(data['class'].value_counts()))

    print("Rows with missing values")
    print(sum(numpy.isnan(data).any(axis=1)))
    print("Attributes with missing values")
    print(len(data.isnull().sum().loc[data.isnull().sum() > 0]))

    # Remove rows with missing data
    cleanData = data[~numpy.isnan(data).any(axis=1)]
    cleanData = cleanData.reset_index(drop=True)  # Required. Otherwise, the index of the rows dropped keep active

    print("Clean data")
    print(cleanData[:])

    # Separating classes (Y) from values (X)
    dataX = cleanData.ix[:, 0:13]
    dataY = cleanData.ix[:, 13]

    # Extract status and standardize product values

    attributes = preprocessing.scale(dataX)
    print("Scaled data")
    print(attributes[:])
    return attributes, dataY


def exercise1(attributes, classes):
    print("Activity 1a")
    # Apply PCA requesting all components (no argument)
    pca = PCA()
    pca.fit(attributes)
    # Varianza explicada acumulada por cada componente
    comultative_variance_ratio = pca.explained_variance_ratio_.cumsum()
    print("cumulative variance ratio", comultative_variance_ratio)

    plt.plot(comultative_variance_ratio)
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()

    for n_components in [2, 3, 8]:
        print(">> Number of principal components:", n_components)
        comulative_variance = comultative_variance_ratio[n_components - 1]
        print("Comulative variance", comulative_variance)
        pca_n = PCA(n_components=n_components)
        pca_n.fit(attributes)
        attributes_pca_n = pca_n.transform(attributes)
        original_from_inverse = pca_n.inverse_transform(attributes_pca_n)
        loss = mean_squared_error(attributes, original_from_inverse)
        print("Information loss:", loss)
        print("Comulative variance + loss", comulative_variance + loss)
        print()

    # Representar los datos en el espacio original con 2 variables:
    fig = plt.figure(1, figsize=(10, 6))
    ax = fig.gca()
    ax.scatter(attributes[0, :], attributes[1, :], marker='o', label='Original')

    # Representar los datos en el espacio 2 PC:
    pca2 = PCA(n_components=12)
    pca2.fit(attributes)
    result = pca2.transform(attributes)
    ax.scatter(result[0,:], result[1,:], marker='+', label='PCA')
    ax.legend()
    plt.show()


def exercise2(attributes, classes):
    print("Activity 2")

    X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.33)

    print(">> kNeighborsClassifier with original data")
    kNeighborsClassifier = KNeighborsClassifier(n_neighbors=3)
    kNeighborsClassifier.fit(X_train, y_train)
    y_pred = kNeighborsClassifier.predict(X_test)
    print("   Accuracy score:", accuracy_score(y_test, y_pred))
    print()

    print(">>  kNeighborsClassifier with PCA data")
    kNeighborsClassifier = KNeighborsClassifier(n_neighbors=3)
    pca = PCA(n_components=8)
    pca.fit(X_train)
    train_data = pca.transform(X_train)
    test_data = pca.transform(X_test)
    kNeighborsClassifier.fit(train_data, y_train)
    predict_pca = kNeighborsClassifier.predict(test_data)
    print("   Accuracy score:", accuracy_score(y_test, predict_pca))


def exercise3(attributes, classes):
    print("Activity 3")

    X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.33)

    classifiers = [
        ["KNeighborsClassifier with 3 neighbors", KNeighborsClassifier(n_neighbors=3)],
        ["KNeighborsClassifier with 4 neighbors", KNeighborsClassifier(n_neighbors=4)],
        ["KNeighborsClassifier with 5 neighbors", KNeighborsClassifier(n_neighbors=5)],
        ["SVM", SVC(kernel="linear", C=0.025)],
        ["Decision Tree", DecisionTreeClassifier(criterion="entropy", max_depth=5)],
        ["AdaBoost", AdaBoostClassifier()],
        ["Gaussian Naive Bayes", GaussianNB()],
    ]
    for classifier_data in classifiers:
        print(">> " + classifier_data[0])
        classifier = classifier_data[1]
        start = now_in_milis()
        classifier.fit(X_train, y_train)
        end = now_in_milis()
        print("   Training time in milliseconds:", end - start)

        start = now_in_milis()
        prediction = classifier.predict(X_test)
        end = now_in_milis()
        print("   Prediction time in milliseconds:", end - start)
        print("   Score:", accuracy_score(y_test, prediction))
        print()


def exercise4(attributes, classes):
    print("Activity 4")

    X_train, X_test, y_train, y_test = train_test_split(attributes, classes, test_size=0.3)
    classifier = SVC(kernel="linear", C=0.025)
    classifier.fit(X_train, y_train)
    prediction = classifier.predict(X_test)

    score = classifier.score(X_test, y_test)
    print("   Default score: ", score)
    precision_macro = precision_score(y_test, prediction, average='macro')
    print("   Precision score: ", precision_macro)

    cross_validators = [
        ['kFold', KFold()],
        ['StratifiedKFold', StratifiedKFold()]
    ]

    for validator_data in cross_validators:
        classifier = SVC(kernel="linear", C=0.025)
        print(">> " + validator_data[0])
        validator = validator_data[1]
        for train_index, test_index in validator.split(attributes, classes):
            X_train, X_test = attributes[train_index], attributes[test_index]
            y_train, y_test = classes[train_index], classes[test_index]
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        print("   Default score: ", score)
        prediction = classifier.predict(X_test)
        precision_macro = precision_score(y_test, prediction, average='macro')
        print("   Precision score: ", precision_macro)


def now_in_milis():
    return time.perf_counter() * 1000


# MAIN

X, y = loadAndPreprocess('../data/processedCleveland.csv')

exercise1(X, y)

exercise2(X, y)

exercise3(X, y)

exercise4(X, y)

print("Fin")