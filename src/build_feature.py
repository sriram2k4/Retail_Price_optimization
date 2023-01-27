import pandas as pd

# revenue
# profit
# profit_margin
# sales_change

class BuildFeatures:
    def __init__(self,df):
        self.df = df
    
    def build_features(self):
        self.df["revenue"] = self.df["price"] * self.df["sales"]
        self.df["profit"] = self.df["revenue"] - self.df["cost"]
        self.df["profit_margin"] = self.df["profit"] / self.df["revenue"]
        self.df["sales_change"] = self.df["sales"].diff()

        return self.df