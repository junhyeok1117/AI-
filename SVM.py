#sklearn_diabetes_dataset_SVM
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes

diabetse = datasets.load_diabetes()
s = svm.SVC(gamma = 0.001)
accuracies = cross_val_score(s, digit.data, digit.target, cv = 100)#cv = k, k가 크면 신뢰도는 향상되지만 실행시간이 증가

print(accuracies)
print("평균 정확률:", accuracies.mean()*100,"%")

