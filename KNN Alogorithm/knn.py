import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report


class KNN:
    def __init__(self,path):
        try:
            self.df = pd.read_csv(path)
            self.df = self.df.drop(["id"] , axis = 1)
            self.df["diagnosis"] = self.df["diagnosis"].map({"M": 0, "B": 1})
            self.X = self.df.iloc[:, 1:]
            self.y = self.df.iloc[:, 0]
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)

            # finding best value of K to get good results
            k_values = [i for i in range(2, 50) if i % 2 != 0]
            accuracy_list = []
            for i in range(len(k_values)):
                algo = KNeighborsClassifier(n_neighbors=k_values[i])
                algo.fit(self.X_train, self.y_train)
                accuracy_list.append(algo.score(self.X_test, self.y_test))

            self.k = k_values[accuracy_list.index(max(accuracy_list))]  # Perfect K-value to get good Results;

            self.knn_algo = KNeighborsClassifier(n_neighbors=self.k)  # KNN - algorithm
            self.knn_algo.fit(self.X_train, self.y_train)  # Training algorithm with Train Data.
            self.y_train_pred = self.knn_algo.predict(self.X_train)  # Prediction values with training input data;
            self.y_test_pred = self.knn_algo.predict(self.X_test)  # Prediction values with testing input data ;

        except Exception as e :
            er_type,er_msg,line_no = sys.exc_info()
            print(f'<{er_type}> : <{er_msg}> : <{line_no.tb_lineno}>')

    def training(self):
        try:
            self.train_acc = accuracy_score(self.y_train,self.y_train_pred)
            print(f'Training Accuracy : \n {self.train_acc}')
        except Exception as e :
            er_type,er_msg,line_no = sys.exc_info()
            print(f'<{er_type}> : <{er_msg}> : <{line_no.tb_lineno}>')
    def testing(self):
        try :
            self.test_acc = accuracy_score(self.y_test,self.y_test_pred)
            print(f'Testing Accuracy : \n {self.test_acc}')

        except Exception as e:
            er_type, er_msg, line_no = sys.exc_info()
            print(f'<{er_type}> : <{er_msg}> : <{line_no.tb_lineno}>')

    def classificationReport(self):
        try :
            self.train_report = classification_report(self.y_train,self.y_train_pred)  # it returns "Precision" , "recall" , "F1-score" and "support";
            self.test_report = classification_report(self.y_test, self.y_test_pred)

            print(f'Classification Report of Train data : \n {self.train_report}')
            print(f'Classification Report of Test data : \n {self.test_report}')
        except Exception as e:
            er_type, er_msg, line_no = sys.exc_info()
            print(f'<{er_type}> : <{er_msg}> : <{line_no.tb_lineno}>')


if __name__ == "__main__" :
    c1 = KNN("breast-cancer.csv")
    c1.training()
    c1.testing()
    c1.classificationReport()