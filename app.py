
import pickle

from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


iris = load_iris()
df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                  columns= iris['feature_names'] + ['target'])

X = df.iloc[:100,[0,2]].values
y = df.iloc[0:100,4].values
y = np.where(y == 0, -1, 1)

class Perceptron():
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
    
    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X,y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update*xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_[1:])+self.w_[0]
    
    def predict(self, X):
        return np.where(self.net_input(X)>=0.0,1,-1)
    
    
# uzycie jak wszsytkie klasy sklearn
ppn = Perceptron(n_iter=20)
ppn.fit(X,y)

### Zapisz swój najlepszy model 
with open('model.pkl', "wb") as picklefile:
    pickle.dump(ppn, picklefile)

# Create a flask
app = Flask(__name__)

# Create an API end point
@app.route('/api/v1.0/predict', methods=['GET'])
def get_prediction():

    # sepal length
    sepal_length = float(request.args.get('sl'))
    petal_length = float(request.args.get('pl'))
    
    features = [sepal_length,
                petal_length]

    # Load pickled model file
    with open('model.pkl',"rb") as picklefile:
        model = pickle.load(picklefile)
        
    # Predict the class using the model
    predicted_class = int(model.predict(features))
    
    # Return a json object containing the features and prediction
    return jsonify({
    'input_features': {
        'sepal_length': sepal_length,
        'petal_length': petal_length
    },
    'predicted_class': predicted_class
})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
