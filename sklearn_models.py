from dataset import generete_data, set_random
from sklearn.svm import SVC
import numpy as np

#set random seed for reproducibility
set_random(42)

#generate data
X_train, y_train, X_test, y_test = generete_data()

#train linear SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
#evaluate model train accuracy
train_accuracy = svm_model.score(X_train, y_train)
print(f"Train Accuracy: {train_accuracy:.4f}")
#evaluate model on test set
accuracy = svm_model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

#train RBF SVM model
svm_model_rbf = SVC(kernel='rbf')
svm_model_rbf.fit(X_train, y_train)
#evaluate model train accuracy
train_accuracy_rbf = svm_model_rbf.score(X_train, y_train)
print(f"Train Accuracy (RBF): {train_accuracy_rbf:.4f}")
#evaluate model on test set
accuracy_rbf = svm_model_rbf.score(X_test, y_test)
print(f"Test Accuracy (RBF): {accuracy_rbf:.4f}")