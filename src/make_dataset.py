import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from config.config import DATA_DIR

warnings.filterwarnings("ignore")

class Ingestor:
    def __init__(self, file_name: str):
        self.file_name = file_name
     
    def load_dataset(self):
        file_path = Path(DATA_DIR, self.file_name)
        df = pd.read_csv(file_path)
        return df

class LabelEncoder:
    def __init__(self,df,cats_col):
        self.df = df
        self.cats_col = cats_col

    def fit(self):
        self.cats_dict = {}
        for col in self.cats_col:
            self.cats_dict[col] = {k:v for v,k in enumerate(self.df[col].unique(),0)}
        return self

    def transform(self):
        for col in self.cats_col:
            self.df[col] = self.df[col].map(self.cats_dict[col])
        return self.df
        
    def fit_transform(self):
        return self.fit().transform()

    def inverse_transform(self):
        #Code for Inverse Transform
        pass

class ProcessData:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def remove_null_values(self):
        df = self.df.copy()
        df = df.dropna()
        return df
