import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
import sys

class Regression:
    def __init__(self,path):
        try:
            self.df = pd.read_csv(path)
            self.df = self.df.drop(["date", "street", "statezip"], axis=1)
            self.unique_city = self.df["city"].unique()
            self.di = {self.unique_city[i] : i for i in range(len(self.unique_city))} # dictionary comprehension
            self.df["city"] = self.df["city"].map(self.di)
            self.df["country"] = self.df["country"].map({"USA" : 0 })
            self.X = self.df.iloc[:,1:]
            self.y = self.df.iloc[:,0]

            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)

            self.reg = LinearRegression()
            self.reg.fit(self.X_train,self.y_train)

        except Exception as e :
            exc_type, exc_value, line_no = sys.exc_info()
            print(f'<{exc_value}> in Line Number <{line_no.tb_lineno}>')

    def training(self):
        try :
            self.y_train_pred = self.reg.predict(self.X_train)
            self.y_train_array = np.array(self.y_train)

            self.loss = sum([(self.y_train_array[i]-self.y_train_pred[i])**2 for i in range(len(self.y_train_array))])
            self.varience = sum([(self.y_train_array[i]-self.y_train_array.mean())**2 for i in range(len(self.y_train_array))])

            self.train_accuracy = 1 - (self.loss/self.varience)
            self.train_Error = (1/len(self.y_train_array)) * self.loss

            print(f'Train Accuracy >>: {self.train_accuracy}')
            print(f'Train Error >>: {self.train_Error}')

        except Exception as e:
            exc_type, exc_value, line_no = sys.exc_info()
            print(f'<{exc_value}> in Line Number <{line_no.tb_lineno}>')

    def testing(self):
        try:
            self.y_test_pred = self.reg.predict(self.X_test)
            self.y_test_array = np.array(self.y_test)

            self.test_loss = sum([(self.y_test_array[i] - self.y_test_pred[i]) ** 2 for i in range(len(self.y_test_array))])
            self.test_varience = sum([(self.y_test_array[i] - self.y_test_array.mean()) ** 2 for i in range(len(self.y_test_array))])

            self.test_accuracy = 1 - (self.test_loss / self.test_varience)
            self.test_Error = (1 / len(self.y_test_array)) * self.test_loss

            print(f'Test Accuracy >>: {self.test_accuracy}')
            print(f'Test Error >>: {self.test_Error}')

        except Exception as e:
            exc_type, exc_value, line_no = sys.exc_info()
            print(f'<{exc_value}> in Line Number <{line_no.tb_lineno}>')


if __name__ == "__main__" :
    c1 = Regression("D:\My Machine Learning\Regression Folder\Reg_Task1\data.csv")
    c1.training()
    c1.testing()