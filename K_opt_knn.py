from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def knn_optimization(k,x_train2,y_train2,x_test,y_test):
    #k = [5,10,20,30,40]
    accuracy_list = []
    precision_list = []
    recall_list = []
    accuracy_list_test = []
    precision_list_test = []
    recall_list_test = []

    for i in k:

        knn = KNeighborsClassifier(n_neighbors = i)
        accuracy = cross_val_score(knn, x_train2,y_train2, cv=5, scoring='accuracy').mean()
        precision = cross_val_score(knn, x_train2,y_train2, cv=5, scoring='precision').mean()
        recall = cross_val_score(knn, x_train2,y_train2, cv=5, scoring='recall').mean()

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)


        accuracy_test = cross_val_score(knn, x_test,y_test, cv=2, scoring='accuracy').mean()
        precision_test = cross_val_score(knn, x_test,y_test, cv=2, scoring='precision').mean()
        recall_test = cross_val_score(knn, x_test,y_test, cv=2, scoring='recall').mean()

        accuracy_list_test.append(accuracy_test)
        precision_list_test.append(precision_test)
        recall_list_test.append(recall_test)
        
    return accuracy_list,precision_list,recall_list,accuracy_list_test,precision_list_test,recall_list_test
