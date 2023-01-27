import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (explained_variance_score, max_error,mean_absolute_error,mean_squared_error,r2_score)

class Evaluate:
    def __init__(self,x, y, model):
        self.model = model
        self.x = x
        self.y = y

    def evaluate(self):
        self.x = sm.add_constant(self.x)
        y_pred = self.model.predict(self.x)
        print(self.y)
        print(y_pred)
        print(f"Mean Absolute Error : {mean_absolute_error(self.y, y_pred)}")
        print(f"Mean Squared Error : {mean_squared_error(self.y, y_pred)}")
        print(f"Root mean Squared Error : {np.sqrt(mean_squared_error(self.y, y_pred))}")