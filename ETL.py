import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network  import MLPClassifier

# graph
import graphviz
from sklearn import tree

class AnalyzeIris:   
    def __init__(self):
        iris = load_iris()
        self.X = iris.data
        self.y = iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=0)
        self.feature_names = iris.feature_names
    
    def Get(self):
        df = pd.DataFrame(data = self.X, columns=self.feature_names)
        df['Label'] = self.y
        return df
    
    def PairPlot(self, cmap='brg'):
        data = self.Get()
        target = data['Label']
        data = data.drop('Label', axis=1)
        pd.plotting.scatter_matrix( data ,c=target,figsize=(15, 15), hist_kwds={'bins': 20}, s=60, alpha=0.8, cmap=cmap)
        plt.show()
        
    def AllSupervised(self, n_neighbors):
        methods = {
            'LogisticRegression': LogisticRegression(max_iter=10000),
            'LinearSVC': LinearSVC(max_iter=10000,dual=True),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=n_neighbors),
            'LinearRegression': LinearRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'MLPClassifier': MLPClassifier(max_iter=10000)
        }
        
        test_score_list = []
        
        for name, model in methods.items():
            print(f"=== {name} ===")
            test_score = cross_val_score(model, self.X_test, self.y_test, cv=5)
            train_score = cross_val_score(model, self.X_train, self.y_train, cv=5)
            test_score_list.append(test_score)
            
            for i in range(5):
                print(f"test_score : {test_score[i]:.3f}   train_score : {train_score[i]:.3f}")

        return 
        
    def GetSupervised(self):
        methods = {
            'LogisticRegression': LogisticRegression(max_iter=10000),
            'LinearSVC': LinearSVC(max_iter=10000,dual=True),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=4),
            'LinearRegression': LinearRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'MLPClassifier': MLPClassifier(max_iter=10000)
        }
        test_score_list = []
        
        for name, model in methods.items():
            test_score = cross_val_score(model, self.X_test, self.y_test, cv=5)
            test_score_list.append(test_score)
        
        model_names = list(methods.keys())
        dict1 = dict(zip(model_names, test_score_list))
        df = pd.DataFrame(data=dict1)
        return df
    
    def BestSupervised(self):
        data = self.GetSupervised().describe()
        best_method = data.loc["mean"].idxmax()
        best_score = data.loc["mean"].max()
        return best_method, best_score
    
    def PlotFeatureImportancesAll(self):
        methods = {
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier()
        }
        for name, model in methods.items():
            model.fit(self.X_train, self.y_train)
            labels = self.feature_names
            importances = model.feature_importances_
            plt.figure(figsize=(10, 6))
            plt.barh(y=range(len(importances )), width=importances )
            plt.yticks(ticks=range(len(labels)), labels=labels)
            plt.xlabel(f'Feature Importance : {name}')
            plt.show()    
            
    def VisualizeDecisionTree(self):
        clf = DecisionTreeClassifier().fit(self.X_train, self.y_train)
        graph = graphviz.Source(tree.export_graphviz(clf, class_names=self.feature_names, filled=True))
        return graph