# SRO6.1-Kzh
# Импорт необходимых библиотек
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Загрузка набора данных (например, набор данных Iris)
data = load_iris()
X = data.data
y = data.target

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание экземпляра случайного леса
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Количество деревьев (n_estimators) можно настраивать

# Обучение случайного леса на тренировочных данных
rf_classifier.fit(X_train, y_train)

# Предсказание на тестовых данных
predictions = rf_classifier.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, predictions)
print("Точность модели случайного леса: {:.2f}%".format(accuracy * 100))
