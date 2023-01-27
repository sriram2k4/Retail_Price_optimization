import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df, test_size=0.2):
    x = df.drop("sales", axis = 1)
    y = df["sales"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    return x_train, x_test, y_train, y_test
