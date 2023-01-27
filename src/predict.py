import statsmodels.api as sm

class Predict:
    def __init__(self, input_data, model):
        self.input_data = input_data
        self.model = model
    
    def predict(self):
        self.input_data = sm.add_constant(self.input_data)
        y_pred = self.model.predict(self.input_data)
        return y_pred