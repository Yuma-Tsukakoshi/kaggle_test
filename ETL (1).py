from sklearn.datasets import load_iris#アイリスのデータセットを読み込む
from sklearn.model_selection import train_test_split#訓練データとテストデータを分割
from IPython.display import display#データフレームの表示
from sklearn.linear_model import LogisticRegression#ロジスティクス回帰
from sklearn.svm import LinearSVC#線形SVC
from sklearn.tree import DecisionTreeClassifier#決定木
from sklearn.neighbors import KNeighborsClassifier#k最近傍法
from sklearn.linear_model import LinearRegression#線形回帰 
from sklearn.ensemble import RandomForestClassifier#ランダムフォレスト
from sklearn.ensemble import GradientBoostingClassifier#勾配ブースティング回帰木
from sklearn.neural_network import MLPClassifier#多層パーセプトロン
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import export_graphviz
from statistics import mean
from sklearn.decomposition import PCA,NMF
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans,AgglomerativeClustering,DBSCAN
from scipy.cluster.hierarchy import dendrogram,ward,linkage,complete,average

#前処理#
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,Normalizer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn
from PIL import Image
import pydotplus#dot言語を扱うためのpythonモジュール
import io

class AnalyzeIris:
    def __init__(self):
        self.iris = load_iris()
        self.X, self.y = self.iris.data, self.iris.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, random_state=0)
        self.feature_names = load_iris().feature_names
        
    def Get(self):
        df_features = pd.DataFrame(data=self.X, columns=self.feature_names)
        df_target = pd.DataFrame(data=self.y, columns=['label'])
        df = pd.concat([df_features, df_target], axis=1)
        #pd.set_option('display.max_rows',150)
        #pd.set_option('display.max_columns',5)
        display(df)
        
    def PairPlot(self, cmap=None):
        iris_dataframe = pd.DataFrame(self.X_train, columns=self.feature_names)
        if cmap is None:
            cmap = "brg"
        else:
            cmap = cmap
        grr = pd.plotting.scatter_matrix(iris_dataframe, c=self.y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=0.8, cmap=cmap)
        plt.show()
        
    def AllSupervised(self, n_neighbors):
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        models = {
            'LogisticRegression': LogisticRegression(max_iter=10000),
            'LinearSVC': LinearSVC(C=10, max_iter=100000),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=n_neighbors),
            'LinearRegression': LinearRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'MLPClassifier': MLPClassifier(max_iter=10000)
        }
        
        #分割間でそれぞれテストデータ，訓練データの精度を出す#
        for model_name, model in models.items():
            print(f"==={model_name}===")
            model.fit(self.X_train,self.y_train)
            test_scores = cross_val_score(model, self.X_test, self.y_test, cv=kfold)
            train_scores = cross_val_score(model, self.X_train, self.y_train, cv=kfold)
            for i in range(5):
                print("test score: {:.3f}    train score: {:.3f}".format(test_scores[i], train_scores[i]))
                
    def GetSupervised(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        models = {
            'LogisticRegression': LogisticRegression(max_iter=10000),
            'LinearSVC': LinearSVC(C=10, max_iter=100000),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=4),
            'LinearRegression': LinearRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'MLPClassifier': MLPClassifier(max_iter=10000)
        }
        test_scores_by_model = []
        for model_name, model in models.items():
            model.fit(self.X_train,self.y_train)
            test_scores = cross_val_score(model, self.X_test, self.y_test, cv=kfold)
            test_scores_by_model.append(test_scores)
        pd.set_option('display.max_rows',8)
        pd.set_option('display.max_columns',8)
        df_model = pd.DataFrame({name: scores for name, scores in zip(models.keys(), test_scores_by_model)})
        return df_model
    
    def BestSurpervised(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        models = {
            'LogisticRegression': LogisticRegression(max_iter=10000),
            'LinearSVC': LinearSVC(C=10, max_iter=100000),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=4),
            'LinearRegression': LinearRegression(),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'MLPClassifier': MLPClassifier(max_iter=10000)
        }
        test_scores_by_model = {}
        for model_name, model in models.items():
            model.fit(self.X_train,self.y_train)
            test_scores = cross_val_score(model, self.X_test, self.y_test, cv=kfold)
            #test_scores = cross_val_score(model, self.X, self.y, cv=kfold)
            test_scores_by_model[model_name]=mean(test_scores)
 
        model_name = max(test_scores_by_model,key=test_scores_by_model.get)
        test_scores_mean_max = test_scores_by_model[model_name]
        return model_name, test_scores_mean_max
        
    def PlotFeatureImportancesAll(self):
        models = {
            'DecisionTreeClassifier': DecisionTreeClassifier(max_depth=4, random_state=0),
            'RandomForestClassifier': RandomForestClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier()
        }
        for model_name, model in models.items():
            model.fit(self.X_train, self.y_train)
            n_features = self.X.shape[1]
            plt.figure()
            plt.barh(range(n_features), model.feature_importances_, align='center')
            plt.yticks(np.arange(n_features), self.feature_names)
            plt.xlabel(f"Feature importance: {model_name}")
        plt.show()

    def VisualizeDecisionTree(self):
        tree = DecisionTreeClassifier(max_depth=4, random_state=0)
        tree.fit(self.X_train,self.y_train)
        out_file = "tree.dot"
        export_graphviz(tree,out_file,class_names=["L","a","b"],feature_names=self.iris.feature_names,impurity=False,filled=True)
        graph = pydotplus.graph_from_dot_file(path=out_file)#"tree.dot"ファイルを読み込みグラフを生成する
        png_data = graph.create_png()#PNG形式の画像データを生成
        binary_data = io.BytesIO(png_data)#PNG形式の画像データをバイナリデータとして保存
        img = Image.open(binary_data)#バイナリデータを画像データに変換
        plt.figure(figsize=(10,10))
        plt.axis("off")
        plt.imshow(img)
        plt.show()
        #L:setosa,a:virsicolor, b:virginica
    
    #練習問題3 データ変換，スケーリング　クラスタリング#
    #5-foldでそれぞれの要素に対するスケーリングとその時のLinearSVCの結果を一覧#
    def PlotScaledData(self):
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        model = LinearSVC(C=10, max_iter=100000)
        
        model.fit(self.X_train,self.y_train)
        test_scores = cross_val_score(model, self.X_test, self.y_test, cv=kfold)
        train_scores = cross_val_score(model, self.X_train, self.y_train, cv=kfold)
    
        scalers = {
            "MinMaxScaler": MinMaxScaler(),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "Normalizer": Normalizer()
        }
        
        features = [
            ("sepal length(cm)","sepal width(cm)"),
            ("sepal width(cm)","petal length(cm)"),
            ("petal length(cm)","petal width(cm)"),
            ("petal width(cm)","sepal length(cm)")
        ]
        
        #それぞれのオリジナルデータに対してスケーリングを行う#
        for fold_idx, (train_index, test_index) in enumerate(kfold.split(self.X_train), 0):
            X_train_fold, X_test_fold = self.X_train[train_index], self.X_train[test_index]
            y_train_fold, y_test_fold = self.y_train[train_index], self.y_train[test_index] 
            #それぞれのfeatureのペアでプロットする(4回)
            for j in range(4):
                i=0
                fig, axes = plt.subplots(1, 5, figsize=(15, 5))
                #1領域のプロット
                for idx, (scaler_name, scaler) in enumerate(scalers.items(),1):
                    X_train_scaled = scaler.fit_transform(X_train_fold)
                    X_test_scaled = scaler.transform(X_test_fold)
                    if i == 0 :
                        print("{:<20}: test score:{:.3f} train score:{:.3f}".format("Original", test_scores[fold_idx], train_scores[fold_idx]))
                        if j==3:
                            axes[i].scatter(self.X_train[:, 3], self.X_train[:, 0],
                                                color=mglearn.cm2(0), label="Training set", s=60)
                            axes[i].scatter(self.X_test[:, 3], self.X_test[:, 0], marker='^',
                                                color=mglearn.cm2(1), label="Test set", s=60) 
                            
                        else:
                            axes[i].scatter(self.X_train[:, j], self.X_train[:, j+1],
                                                color=mglearn.cm2(0), label="Training set", s=60)
                            axes[i].scatter(self.X_test[:, j], self.X_test[:, j+1], marker='^',
                                                color=mglearn.cm2(1), label="Test set", s=60) 
                            
                        axes[i].legend(loc="upper left")
                        axes[i].set_title("Original Data")
                        i=1
        
                    if idx == 1:
                        model.fit(X_train_scaled, y_train_fold)  # モデルをスケーリングされたデータで再度トレーニング
                        print("{:<20}: test score:{:.3f} train score:{:.3f}".format(scaler_name,model.score(X_test_scaled,                y_test_fold),model.score(X_train_scaled, y_train_fold)))
                        if j==3:
                            axes[idx].scatter(X_train_scaled[:, 3], X_train_scaled[:, 0],
                                                color=mglearn.cm2(0), label="Training set", s=60)
                            axes[idx].scatter(X_test_scaled[:, 3], X_test_scaled[:, 0], marker='^',
                                                color=mglearn.cm2(1), label="Test set", s=60) 
                        else:
                            axes[idx].scatter(X_train_scaled[:, j], X_train_scaled[:, j+1],
                                                color=mglearn.cm2(0), label="Training set", s=60)
                            axes[idx].scatter(X_test_scaled[:, j], X_test_scaled[:, j+1], marker='^',
                                                color=mglearn.cm2(1), label="Test set", s=60) 
                    
                        axes[idx].set_title("MinMaxScaler")

                    elif idx > 1:
                        model.fit(X_train_scaled, y_train_fold)  # モデルをスケーリングされたデータで再度トレーニング
                        print("{:<20}: test score:{:.3f} train score:{:.3f}".format(scaler_name, model.score(X_test_scaled, y_test_fold), model.score(X_train_scaled, y_train_fold)))
                        if j==3:
                            axes[idx].scatter(X_train_scaled[:, 3], X_train_scaled[:, 0],
                                                color=mglearn.cm2(0), label="Training set", s=60)
                            axes[idx].scatter(X_test_scaled[:, 3], X_test_scaled[:, 0], marker='^',
                                                color=mglearn.cm2(1), label="Test set", s=60) 
                        else:
                            axes[idx].scatter(X_train_scaled[:, j], X_train_scaled[:, j+1],
                                                color=mglearn.cm2(0), label="Training set", s=60)
                            axes[idx].scatter(X_test_scaled[:, j], X_test_scaled[:, j+1], marker='^',
                                                color=mglearn.cm2(1), label="Test set", s=60) 
                        axes[idx].set_title(scaler_name)
                        
                    for ax in axes:
                        ax.set_xlabel(features[j][0],size=10)
                        ax.set_ylabel(features[j][1],size=10)
                plt.tight_layout()
                plt.show()
                print("="*130)
    
    def PlotFeatureHistgram(self):
        fig,axes = plt.subplots(4,1,figsize=(10,15))
        setosa = self.X[self.y==0]
        versicolor = self.X[self.y==1]
        virginica = self.X[self.y==2]
        ax = axes.ravel()

        for i in range(4):
            _,bins = np.histogram(self.X[:,i],bins=60)#_:ビンごとのデータ数，bins:使用されたビンの範囲
            ax[i].hist(setosa[:,i],bins=bins,color=mglearn.cm3(0),alpha=.5)
            ax[i].hist(versicolor[:,i],bins=bins,color=mglearn.cm3(1),alpha=.5)
            ax[i].hist(virginica[:,i],bins=bins,color=mglearn.cm3(2),alpha=.5)
            ax[i].set_title(self.feature_names[i])
            ax[i].set_yticks(())
            
        ax[0].set_xlabel("Feature magnitude")
        ax[0].set_ylabel("Frequency")
        ax[0].legend(["setosa","versicolor","virginica"],loc="best")
        fig.tight_layout()
        plt.show()
        
    def PlotPCA(self,n_components):
        scaler = StandardScaler()
        scaler.fit(self.X)
        X_scaled = scaler.transform(self.X)
        X_pca = PCA(n_components = n_components)
        X_pca_transform = X_pca.fit_transform(X_scaled)
        plt.figure(figsize=(8,8))
        mglearn.discrete_scatter(X_pca_transform[:,0],X_pca_transform[:,1],self.y)
        plt.legend(self.iris.target_names,loc = "lower right")
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.show()
        
        plt.matshow(X_pca.components_,cmap='viridis')
        plt.yticks([0,1],["First component","Second component"])
        plt.colorbar()
        plt.xticks(range(len(self.feature_names)),
                   self.feature_names,rotation=60,ha='left')
        plt.xlabel("Feature")
        plt.ylabel("Principal components")
        plt.show()
        df_pca = pd.DataFrame(data=X_pca_transform, columns=[0,1])
        X_scaled = pd.DataFrame(data=X_scaled,columns=[self.feature_names])
        return X_scaled,df_pca,X_pca
    
    def PlotNMF(self,n_components):
        #必要あるのかMinMaxScalerで非負値にする必要あり？#
        scaler = StandardScaler()
        scaler.fit(self.X)
        X_scaled = scaler.transform(self.X)
        X_nmf = NMF(n_components = n_components,random_state=0)
        X_nmf_transform = X_nmf.fit_transform(self.X)
        plt.figure(figsize=(8,8))
        mglearn.discrete_scatter(X_nmf_transform[:,0],X_nmf_transform[:,1],self.y)
        plt.legend(self.iris.target_names,loc = "upper right")
        plt.xlabel("First principal component")
        plt.ylabel("Second principal component")
        plt.show()
        
        plt.matshow(X_nmf.components_,cmap='viridis')
        plt.yticks([0,1],["First component","Second component"])
        plt.colorbar()
        plt.xticks(range(len(self.feature_names)),
                   self.feature_names,rotation=60,ha='left')
        plt.xlabel("Feature")
        plt.ylabel("Principal components")
        plt.show()
        df_nmf = pd.DataFrame(data=X_nmf_transform, columns=[0,1])
        X_scaled = pd.DataFrame(data=X_scaled,columns=[self.feature_names])
        return X_scaled,df_nmf,X_nmf
        
    def PlotTSNE(self):
        tsne = TSNE(random_state=42)
        #fitではなくfit_transform
        iris_tsne = tsne.fit_transform(self.X)
        plt.figure(figsize=(7,7))
        plt.ylim(iris_tsne[:,0].min(),iris_tsne[:,0].max())
        plt.xlim(iris_tsne[:,1].min(),iris_tsne[:,1].max())
        for i in range(len(self.X)):
            #散布図を数字でプロット#
            plt.text(iris_tsne[i,1],iris_tsne[i,0],str(self.y[i]))
        plt.xlabel("t-SNE feature 0")
        plt.ylabel("t-SNE feature 1")
        plt.show()
        
    def PlotKMeans(self):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(self.X)
        k_predict = kmeans.predict(self.X)
        print(f"KMeans法で予測したラベル:\n{k_predict}")
        #クラスターごとに異なるマーカーでプロット
        markers = ['o', '*', 'D']  # 使用するマーカースタイルをリストに定義
        for i in range(3):
            plt.scatter(
                self.X[kmeans.labels_ == i, 2], self.X[kmeans.labels_ == i, 3],
                marker=markers[i], edgecolor='black', s=100)
        plt.scatter(
        kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
        marker='^', edgecolor='black', s=150, c='white' ) # 三角形のマーカーの色を白に設定
        plt.show()
        
        print(f"実際のラベル:\n{self.y}")
        plt.figure()
        for i in range(3):
            plt.scatter(
                self.X[self.y == i, 2], self.X[self.y == i, 3],
                marker=markers[i], edgecolor='black', s=100)
            
        plt.scatter(
        kmeans.cluster_centers_[:, 2], kmeans.cluster_centers_[:, 3],
        marker='^', edgecolor='black', s=100, c='white')  # 三角形のマーカーの色を白に設定
        plt.show()
    
    def PlotDendrogram(self, truncate=None):
        linkage_array = ward(self.X)
        if truncate == True:
            plt.figure(figsize=(15,5)) 
            dendrogram(linkage_array, truncate_mode='lastp', p=10)  # 上位のクラスタのみを表示（ここでは上位10個）
        elif truncate == None:
            plt.figure(figsize=(13,6)) 
            dendrogram(linkage_array)
            ax = plt.gca()
            bounds = ax.get_xbound()
            ax.plot(bounds, [10, 10], '--', c='k')
            ax.plot(bounds, [5.5, 5.5], '--', c='k')
            ax.text(bounds[1], 10, '3 clusters', va='center', fontdict={'size': 15})
            ax.text(bounds[1], 5.5, '4 clusters', va='center', fontdict={'size': 15})
            
        plt.xlabel("Sample")
        plt.ylabel("Cluster distance")
        plt.show()
        #左のクラスタから2つ選ばれてしまう．しかし，自動的に選択しているため不可能？#
    
    def PlotDBSCAN(self,scaling=None,eps=None,min_samples=None):
        
        if(scaling==True and eps is not None and min_samples is not None):
            dbscan = DBSCAN(eps=eps,min_samples=min_samples)
            scaler = StandardScaler()
            scaler.fit(self.X)
            X_scaled = scaler.transform(self.X)
            clusters = dbscan.fit_predict(X_scaled)
            print(f"Cluster Memberships:\n{clusters}")
            unique_labels = np.unique(clusters)
            for label in unique_labels:
                X_scaled = self.X[np.where(clusters==label)] 
                plt.scatter(X_scaled[:,2],X_scaled[:,3],label=label)
                
            ##plt.scatter(X_scaled[:,2],X_scaled[:,3],c=clusters,cmap=mglearn.cm3,s=60)
            
        else:
            dbscan = DBSCAN(eps=1.5,min_samples=3)
            clusters = dbscan.fit_predict(self.X)
            
            print(f"Cluster Memberships:\n{clusters}")
            unique_labels = np.unique(clusters)
            for label in unique_labels:
                X = self.X[np.where(clusters==label)] 
                plt.scatter(X[:,2],X[:,3],label=label)
            ##plt.scatter(self.X[:,2],self.X[:,3],c=clusters,cmap=mglearn.cm2,s=60)
       
        plt.xlabel("Feature 2")
        plt.ylabel("Feature 3")
        plt.show()
        