import tensorflow as tf
import numpy as np
from pre_process import data_get, get_sclar
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler as ss

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


X_train, X_test, y_train, y_test = data_get()


ann = tf.keras.Sequential()


ann.add(tf.keras.layers.Dense(units=8, activation="relu"))
ann.add(tf.keras.layers.Dense(units=8, activation="relu"))
ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))


ann.compile(optimizer="rmsprop", loss="binary_crossentropy",
            metrics=["accuracy"])


ann.fit(X_train, y_train, batch_size=32, epochs=75)


y_pred = ann.predict(X_test)
binary_pred = np.round(ann.predict(X_test)).astype(int)

print("Results for Binary Model")
print(accuracy_score(y_test, binary_pred))
print(classification_report(y_test, binary_pred))

sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


def predict_ans(data):
    sc = ss()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    final_input = get_sclar(data)
    output = ann.predict(final_input)[0]
    return output
