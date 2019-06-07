
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


data = datasets.load_iris()


model = GaussianNB()

model.fit(data.data, data.target)

print(model)

# make predictions

expected = data.target

predicted = model.predict(data.data)


print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))
