
import pandas as pd

from sklearn import datasets, linear_model

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt


columns = "age sex bmi map tc ldl hdl tch ltg glu".split()

diabetes = datasets.load_diabetes()

df = pd.DataFrame(diabetes.data, columns=columns)

y = diabetes.target # define the target variable (dependent variable) as y

X_train, X_test, y_train, y_test = train_test_split(df, y,test_size=0.3)

print (X_train.shape, y_train.shape)

print (X_test.shape, y_test.shape)

lm = linear_model.LinearRegression()

model = lm.fit(X_train, y_train)

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
