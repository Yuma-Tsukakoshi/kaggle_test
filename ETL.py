import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


class AnalyzeIris:    
    def Get(self):
        data = load_iris()
        data = pd.DataFrame(data.data, columns=data.feature_names)
        return  data
    
    def GetColumns(self, data):
        return data.target