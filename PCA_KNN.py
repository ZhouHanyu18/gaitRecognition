from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

def PCA_KNN(x_train, x_test, y_train, y_test):
    x_train = x_train.reshape((-1, 50 * 6))
    x_test = x_test.reshape((-1, 50 * 6))
    pca = PCA()
    pca_fit = pca.fit(x_train)
    x_train_pca = pca_fit.transform(x_train)
    x_test_pca = pca_fit.transform(x_test)
    print("x_train_pca.shape: ", x_train_pca.shape)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train_pca, y_train)
    y_predict = knn.predict(x_test_pca)
    score = knn.score(x_test_pca, y_test, sample_weight=None)
    print("acc = {:.2%}".format(score))
    print(classification_report(y_test, y_predict))