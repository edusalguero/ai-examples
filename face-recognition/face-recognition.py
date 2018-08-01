import os
import warnings

import matplotlib.pyplot as plt
from keras import backend as K, Model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import time

DEBUG = True
warnings.filterwarnings("ignore", category=DeprecationWarning)  # Avid Deprecation Warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Avoid TensorFlow warnings

K.set_image_dim_ordering('th')

output_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.path.sep + 'output/'

print(output_path)
def debug_log(string):
    if DEBUG:
        print(string)


# noinspection SpellCheckingInspection
def load_data_set():
    lfw_people = fetch_lfw_people(min_faces_per_person=200, resize=1, color=False, funneled=True)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    x = lfw_people.data
    n_features = x.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    names = lfw_people.target_names
    n_classes = names.shape[0]

    debug_log("Total dataset size:")
    debug_log("n_samples: %d" % n_samples)
    debug_log("n_features: %d" % n_features)
    debug_log("n_classes: %d" % n_classes)

    # #############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

    debug_log("There are %d faces" % x_train.shape[0])
    return (x_train, x_test, y_train, y_test), n_classes, names, h, w


# noinspection SpellCheckingInspection
def prepare_trainset_for_cnn(x_train, x_test, y_train, y_test, image_h, image_w):
    # Dimensionar las imagenes en vectores
    x_train = x_train.reshape(x_train.shape[0], 1, image_h, image_w).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 1, image_h, image_w).astype('float32')

    # Normalizar las imagenes de escala de grises 0-255 (valores entre 0-1):
    x_train = x_train / 255
    x_test = x_test / 255

    return x_train, x_test, y_train, y_test


# noinspection SpellCheckingInspection
def eigenfaces_pca_plus_svm(train_test_splits, n_labels, target_names, n_components=10,
                            print_cumulative_variance=False, print_confusion_matrix=False):
    (X_train, X_test, y_train, y_test) = train_test_splits

    # #############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    debug_log("PCA components %d" % n_components)

    pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
    pca.fit(X_train)
    # Projecting the input data on the eigenfaces orthonormal basis
    x_train_pca = pca.transform(X_train)
    x_test_pca = pca.transform(X_test)

    if print_cumulative_variance:
        cumulative_variance = pca.explained_variance_ratio_.cumsum()
        plt.plot(cumulative_variance)
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance')
        plt.axhline(y=0.95, color='r', linestyle='-', xmax=n_components, linewidth=1)
        plt.savefig(output_path + 'eigengfaces-pca-' + str(n_components) + '-variance.png')

    # #############################################################################
    # Train a SVM classification model
    # Fitting the classifier to the training set
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(x_train_pca, y_train)

    # print("Best estimator found by grid search:")
    debug_log("Best estimator")
    debug_log(clf.best_estimator_)

    # #############################################################################
    # Quantitative evaluation of the model quality on the test set
    if print_confusion_matrix:
        # print("Predicting people's names on the test set")
        y_pred = clf.predict(x_test_pca)
        debug_log(classification_report(y_test, y_pred, target_names=target_names))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=range(n_labels)))

    return cross_val_score(clf.best_estimator_, x_test_pca, y_test, cv=10)


# noinspection SpellCheckingInspection
def cnn_classifier(train_test_splits, num_classes, image_h, image_w):
    (x_train, x_test, y_train, y_test) = train_test_splits
    x_train, x_test, y_train, y_test = prepare_trainset_for_cnn(x_train, x_test, y_train, y_test, image_w=image_w,
                                                                image_h=image_h)
    # Codificar las etiquetas de clase en formato de vectores categoricos con diez posiciones:
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # Capa de entrada:
    model = Sequential()
    # Primera capa (Convolucional):
    model.add(Conv2D(32, (5, 5), input_shape=(1, image_h, image_w), activation='relu'))
    # Tercera capa (agrupamiento):
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Cuarta capa (regularizacion)
    model.add(Dropout(0.2))
    # Quinta capa (redimensionamiento):
    model.add(Flatten())
    # Sexta capa (completamente conectada)
    model.add(Dense(128, activation='relu', use_bias=False))
    # Capa de salida (softmax):
    model.add(Dense(num_classes, activation='softmax'))
    # Compilar el modelo y especificar metodo y metrica de optimizacion:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Ajustar el modelo utilizando los datos de entrenamiento y validacion con los de test:
    if DEBUG:
        verbose = 2
    else:
        verbose = 0

    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=200, verbose=verbose)

    # Evaluacion del modelo utilizando los datos de test:
    score = model.evaluate(x_test, y_test, verbose=0)
    return score[1]


# noinspection SpellCheckingInspection
def deepfaces_plus_svm(train_test_splits, num_classes, image_h, image_w, target_names):
    (x_train, x_test, y_train, y_test) = train_test_splits
    x_train_for_cnn, x_test_for_cnn, y_train_for_cnn, y_test_for_cnn = prepare_trainset_for_cnn(x_train, x_test,
                                                                                                y_train, y_test,
                                                                                                image_w=image_w,
                                                                                                image_h=image_h)
    # Capa de entrada:
    model = Sequential()
    # Primera capa (Convolucional):
    model.add(Conv2D(32, (5, 5), input_shape=(1, image_h, image_w), activation='relu'))
    # Tercera capa (agrupamiento):
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Cuarta capa (regularizacion)
    model.add(Dropout(0.2))
    # Quinta capa (redimensionamiento):
    model.add(Flatten())
    # Sexta capa (completamente conectada)
    model.add(Dense(128, activation='relu', name='f'))
    # Capa de salida (softmax):
    model.add(Dense(num_classes, activation='softmax'))

    cnn = Model(inputs=model.inputs, outputs=model.get_layer('f').output)

    x_pred_cnn = cnn.predict(x_train_for_cnn)
    x_test_cnn = cnn.predict(x_test_for_cnn)

    print(y_train.shape)
    # Train a SVM classification model
    # Fitting the classifier to the training set
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    classifier = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    classifier.fit(x_pred_cnn, y_train)
    debug_log("Best estimator")
    debug_log(classifier.best_estimator_)

    y_pred = classifier.predict(x_test_cnn)
    debug_log(classification_report(y_test, y_pred, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=range(n_labels)))

    return cross_val_score(classifier.best_estimator_, x_test_cnn, y_test, cv=10)


# ==============> Run face recognition techniques
print("\n\n#### Execute face recognition techniques #### \n")
train_test, n_labels, target_names, image_height, image_with = load_data_set()

print("===== Exercise 1: 'Eigenfaces' + SVM =====")
# Exercise 1
ini = time.process_time()
scores = eigenfaces_pca_plus_svm(train_test, n_labels=n_labels, target_names=target_names)
print("==> Accuracy  (PCA 10): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("    Time %0.2f\n" % float(time.process_time() - ini))

ini = time.process_time()
scores = eigenfaces_pca_plus_svm(train_test, n_labels=n_labels, target_names=target_names, n_components=100)
print("==> Accuracy (PCA 100): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("    Time %0.2f\n" % float(time.process_time() - ini))

ini = time.process_time()
scores = eigenfaces_pca_plus_svm(train_test, n_labels=n_labels, target_names=target_names, n_components=400,
                                 print_cumulative_variance=True, print_confusion_matrix=True)
print("==> Accuracy (PCA 400): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("    Time %0.2f\n" % float(time.process_time() - ini))

# Exercise 2
ini = time.process_time()
print("\n===== Exercise 2: CNN classification =====")
accuracy = cnn_classifier(train_test, num_classes=n_labels, image_h=image_height, image_w=image_with)
print("==> Accuracy: %0.2f" % accuracy)
print("    Time %0.2f\n" % float(time.process_time() - ini))

# Exercise 3
ini = time.process_time()
print("\n===== Exercise 3: 'Deepfaces' + SVM =====")
scores = deepfaces_plus_svm(train_test, num_classes=n_labels, image_h=image_height, image_w=image_with,
                            target_names=target_names)
print("==> Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("    Time %0.2f\n" % float(time.process_time() - ini))
