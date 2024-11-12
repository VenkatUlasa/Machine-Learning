import sys

import numpy as np
import pandas as pd
import sklearn
from scipy.constants import precision
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


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
            train_data = self.X_train.copy()
            train_data["Y_Train"] = self.y_train
            train_data["Y_Train_Pred"] = self.y_train_pred
            tp = 0
            tn = 0
            fp = 0
            fn = 0

            for i in train_data.index:
                a = train_data["Y_Train"][i]
                b = train_data["Y_Train_Pred"][i]
                if (a, b) == (1, 1):
                    tp += 1
                elif (a, b) == (0, 0):
                    tn += 1
                elif (a, b) == (0, 1):
                    fp += 1
                elif (a, b) == (1, 0):
                    fn += 1

            self.train_acc = (tp+tn) /(tp+tn+fp+fn)
            print(f'Train Accuracy : {self.train_acc}')

        except Exception as e :
            er_type,er_msg,line_no = sys.exc_info()
            print(f'<{er_type}> : <{er_msg}> : <{line_no.tb_lineno}>')

    def testing(self):
        try :
            self.test_acc = accuracy_score(self.y_test,self.y_test_pred)
            print(f'Test Accuracy : {self.test_acc}')

        except Exception as e:
            er_type, er_msg, line_no = sys.exc_info()
            print(f'<{er_type}> : <{er_msg}> : <{line_no.tb_lineno}>')

    def classification_Report_train(self):
        try :
            c_m = confusion_matrix(self.y_train,self.y_train_pred).ravel()
            tn,fp,fn,tp = c_m

            pre = tp/(tp+fp)  # precision
            rec = tp/(tp+fn)    # recall
            f1_score = 2 * ((pre*rec)/(pre+rec))

            print(f'Train_precision : {pre}')
            print(f'Train recall : {rec}')
            print(f'Train F1-score : {f1_score}')

        except Exception as e:
            er_type, er_msg, line_no = sys.exc_info()
            print(f'<{er_type}> : <{er_msg}> : <{line_no.tb_lineno}>')

    def classification_Report_test(self):
        try:
            c_m = confusion_matrix(self.y_test, self.y_test_pred).ravel()
            tn, fp, fn, tp = c_m

            pre = tp / (tp + fp)  # precision
            rec = tp / (tp + fn)  # recall
            f1_score = 2 * ((pre * rec) / (pre + rec))

            print(f'Test_precision : {pre}')
            print(f'Test recall : {rec}')
            print(f'Test F1-score : {f1_score}')

        except Exception as e:
            er_type, er_msg, line_no = sys.exc_info()
            print(f'<{er_type}> : <{er_msg}> : <{line_no.tb_lineno}>')

if __name__ == "__main__" :
    c1 = KNN("breast-cancer.csv")
    c1.training()
    c1.testing()
    c1.classification_Report_train()
    c1.classification_Report_test()