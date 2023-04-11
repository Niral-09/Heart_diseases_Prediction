import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as ss

# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action="ignore", category=FutureWarning)


df = pd.read_csv("cleveland.csv", header=None)

df.columns = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]


df.isnull().sum()


df["target"] = df.target.map({0: 0, 1: 1, 2: 1, 3: 1, 4: 1})
df["sex"] = df.sex.map({0: "female", 1: "male"})
df["thal"] = df.thal.fillna(df.thal.mean())
df["ca"] = df.ca.fillna(df.ca.mean())
df["sex"] = df.sex.map({"female": 0, "male": 1})


sns.set_context(
    "paper",
    font_scale=2,
    rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20},
)
sns.catplot(
    kind="count", data=df, x="age", hue="target",
    order=df["age"].sort_values().unique()
)
plt.title("Variation of Age for each target class")
plt.show()


# barplot of age vs sex with hue = target
sns.catplot(kind="bar", data=df, y="age", x="sex", hue="target")
plt.title("Distribution of age vs sex with the target class")
plt.show()


plt.figure(figsize=(20, 12))
sns.set_context("notebook", font_scale=1.3)
sns.heatmap(df.corr(), annot=True, linewidth=2)
plt.tight_layout()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=0)

sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# y_test = sc.transform(y_test)
# y_train = sc.fit_transform(X_train)


def data_get():
    return X_train, X_test, y_train, y_test


def get_sclar(data):
    final_input = sc.transform(np.array(data).reshape(1, -1))
    return final_input
