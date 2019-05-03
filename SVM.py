from sklearn import svm


def SVM(x_train, x_test, y_train, y_test):
    x_train = x_train.reshape((-1, 50 * 6))
    x_test = x_test.reshape((-1, 50 * 6))
    model_svc = svm.SVC()
    model_svc.fit(x_train, y_train)
    print(model_svc.score(x_test, y_test))