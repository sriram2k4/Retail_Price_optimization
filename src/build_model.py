import statsmodels.api as sm

class ModelBuilder:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def build_model(self):
        self.x = sm.add_constant(self.x)
        model = sm.OLS(self.y, self.x).fit()
        return model