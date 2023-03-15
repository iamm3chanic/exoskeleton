import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

data = pd.read_csv('dataset.csv')

# Проверяем, всё ли правильно загрузилось

print(data.head(5))
# ".iloc" принимает row_indexer, column_indexer
X = data.iloc[:, :-1].values
# Теперь выделим нужный столбец
y = data['result']
# test_size показывает, какой объем данных нужно выделить для тестового набора
# Random_state — просто сид для случайной генерации
# Этот параметр можно использовать для воссоздания определённого результата:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18, random_state=17)

print(X_train)
print(y_train)

# метод опорных векторов 
SVC_model = SVC()
# метод k-ближайших соседей
# В KNN-модели нужно указать параметр n_neighbors
KNN_model = KNeighborsClassifier(n_neighbors=10)
# логистическая регрессия
logisticRegr = LogisticRegression()

SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
logisticRegr.fit(X_train, y_train)

SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)
LogReg_prediction = logisticRegr.predict(X_test)

# Оценка точности — простейший вариант оценки работы классификатора
print("accuracy_svc =", accuracy_score(SVC_prediction, y_test))
print("accuracy_knn =", accuracy_score(KNN_prediction, y_test))
print("accuracy_logreg =", accuracy_score(LogReg_prediction, y_test))
# Но матрица неточности и отчёт о классификации дадут больше информации о производительности
print("confusion_matrix SVC:\n", confusion_matrix(SVC_prediction, y_test))
print("confusion_matrix KNN:\n", confusion_matrix(KNN_prediction, y_test))
print("confusion_matrix LogReg:\n", confusion_matrix(LogReg_prediction, y_test))

print("classification_report SVC: \n", classification_report(SVC_prediction, y_test))
print("classification_report KNN: \n", classification_report(KNN_prediction, y_test))
print("classification_report LogReg: \n", classification_report(LogReg_prediction, y_test))
