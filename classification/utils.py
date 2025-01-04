import numpy as np
from pyomo.environ import *

class Custom_SVM():
    # Custom linear Support Vector Machine (SVM)
    def __init__(self, w, b):
        self.w = w
        self.b = b

    def fit(self, X, y):
        # Fit the model using the training data
        n, d = X.shape
        # Create a binary integer program
        model = ConcreteModel()
        model.alpha = Var(range(n), within=Binary)
        model.w = Param(range(d), initialize=self.w)
        model.b = Param(initialize=self.b)
        model.X = Param(range(n), range(d), initialize=lambda model, i, j: X[i, j])
        model.y = Param(range(n), initialize=lambda model, i: y[i])
        model.C = Param(initialize=1)
        model.obj = Objective(expr=sum(model.alpha[i] for i in range(n)), sense=minimize)
        model.constraint = ConstraintList()
        for i in range(n):
            model.constraint.add(model.y[i] * (sum(model.w[j] * model.X[i, j] for j in range(d)) + model.b) >= 1 - model.C * (1 - model.alpha[i]))
        SolverFactory('glpk').solve(model)
        # Extract the support vectors
        support_vectors = np.array([X[i] for i in range(n) if model.alpha[i].value == 1])
        # Compute the bias term
        support_vector = support_vectors[0]
        self.b = y[np.argmin(np.dot(support_vectors, self.w.T))]
        return self

    def predict(self, X):
        # Returns the sign of the decision function
        return np.dot(X, self.w.T) + self.b > 0
    
    def score(self, X, y):
        # Returns the accuracy of the model
        return np.mean(self.predict(X) == y)
    
    def empirical_risk(self, X, y):
        # Returns the empirical risk of the model
        return 1 - self.score(X, y)
    
    def decision_function(self, X):
        # Returns the decision function
        return np.sign(np.dot(X, self.w.T) + self.b)
    
