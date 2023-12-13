import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from kmeans_feature_imp import KMeansInterp
from sklearn.metrics.pairwise import cosine_similarity


class PandasSimpleImputer(SimpleImputer):
    """A wrapper around `SimpleImputer` to return data frames with columns.
    """

    def fit(self, X, y=None):
        self.columns = X.columns
        return super().fit(X, y)

    def transform(self, X):
        return pd.DataFrame(super().transform(X), columns=self.columns)


class Data:
    def __init__(self, filename, year=None, target=None):
        self.file = filename
        self.df = pd.read_csv(filename)
        self.X = self.df
        self.findf = self.df
        self.target = ""
        self.imputer = SimpleImputer(strategy="mean")
        self.encoder = OrdinalEncoder()
        self.categorical = ['Country', 'Region', 'SubRegion']
        

        if year:
            self.X = self.get_year(year)

        if target:
            self.set_prediction(target)

    def get_year(self, year):
        grouped_data = self.X.groupby(['Year'])
        return grouped_data.get_group(year)

    def set_prediction(self, attribute):
        self.target = attribute
        self.X = self.findf.drop(columns=[self.target])
        self.y = self.findf[self.target]

    def pre_process(self):

        # Handling missing values

        print(len(self.X.index))
        self.X = self.X.replace(0, np.NaN)
        # Dropping duplicate columns
        to_drop = ['biodiesel_cons_pj', 'biodiesel_prod_pj', 'biofuels_cons_ej', 'biofuels_cons_pj', 'biofuels_prod_kbd', 'biofuels_prod_pj', 'biogeo_twh', 'biogeo_twh_net',
                   'co2_combust_mtco2', 'co2_combust_pc', 'ethanol_cons_pj', 'ethanol_prod_pj', 'gascons_bcfd', 'gascons_bcm', 'gasflared_bcm', 'gasprod_bcfd', 'gasprod_bcm',
                   'hydro_twh', 'hydro_twh_net', 'nuclear_twh', 'nuclear_twh_net', 'oilcons_ej', 'oilcons_mt', 'oilprod_mt', 'primary_eintensity', 'primary_ej_pc', 'ren_power_twh',
                   'ren_power_twh_net', 'solar_twh', 'solar_twh_net', 'wind_twh', 'wind_twh_net']

        self.X = self.X.drop(columns=to_drop)
        self.handle_nulls()

        print(len(self.X.index))
        try:
            self.y = self.y.to_numpy()
        except:
            print(
                "Skipping preprocessing of y as it does not exists or results in some error")

    def list_features(self):
        return self.X.columns

    def show_nulls(self):
        self.X.isna().sum()[self.X.isna().sum() > 0].plot(kind='bar')
        plt.ylabel('Null Values')
        plt.show()

    def handle_nulls(self):
        # dropping columns with high missing values
        to_drop = []
        self.ctr = self.X['Country'].values
        for cols in self.X.columns:
            perc = (self.X[cols].isna().sum())/(len(self.X.index))*100
            # print(cols,perc)
            if (perc >= 50):
                to_drop.append(cols)
                # print(cols,perc)
        # print(len(self.X.index),self.X.isna().sum())
        self.X = self.X.drop(columns=to_drop)
        for cols in self.X.columns:
            if cols in self.categorical:
                self.X[cols] = self.X[cols].astype('category').cat.codes

        self.X = PandasSimpleImputer().fit_transform(self.X)

    def show_outliers(self, column):
        try:
            sns.boxplot(x=self.X[column])
            plt.show()
        except:
            print("There was some error while accessing the given column")

    def IQR(self):
        temp1 = self.X.copy()
        temp2 = self.X.copy()
        Q1 = (temp1).quantile(0.25)
        Q3 = (temp2).quantile(0.75)
        iqr = Q3-Q1
        return (Q1, Q3, iqr)

    def remove_outliers(self, col):
        prev = len(self.X.index)
        Q1 = self.X[col].quantile(0.25)
        Q3 = self.X[col].quantile(0.75)
        IQR = Q3 - Q1

        # identify outliers
        threshold = 1.5
        outliers = self.X[(self.X[col] < Q1 - threshold * IQR)
                          | (self.X[col] > Q3 + threshold * IQR)]
        self.X = self.X.drop(outliers.index)
        cntt = []
        # print(self.ctr)
        for i in range(len(self.ctr)):
            if i not in outliers.index:
                cntt.append(self.ctr[i])
        self.ctr = cntt
        print(f"{prev-len(self.X.index)} outliers were removed")

    def remove_all_outliers(self):
        Q1, Q3, IQR = self.IQR()
        prev = len(self.X.index)
        self.X = self.X[~((self.X < (Q1 - 1.5 * IQR)) |
                          (self.X > (Q3 + 1.5 * IQR))).any(axis=1)]
        print(f"{prev-len(self.X.index)} outliers were removed")

    def heat_map(self):
        c = self.X.corr()
        sns.heatmap(c, cmap='BrBG', annot=True)
        plt.show()


class Model:
    def __init__(self, clusters, Data):
        self.X = Data.X[['Country', 'co2_mtco2', 'elect_twh',
                         'oilcons_kbd', 'coalcons_ej', 'primary_ej']]

        self.cntr = Data.ctr
        self.X = self.X.drop(columns=['Country'])
        self.data = self.X
        self.X = StandardScaler().fit_transform(Data.X)
        self.clusters = clusters

    def elbow(self):
        cost = []
        for i in range(1, self.clusters):
            kmean = KMeans(i, verbose=True)
            kmean.fit(self.X)
            cost.append(kmean.inertia_)
        plt.plot(cost, 'bx-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Cost')
        plt.show()

    def fit(self):
        # kmean = KMeans(self.clusters,verbose=True)
        # kmean.fit(self.X)
        # labels = kmean.labels_

        # plt.show()
        kms = KMeansInterp(
            n_clusters=self.clusters,
            ordered_feature_names=self.data.columns.tolist(),
            feature_importance_method='wcss_min'
        ).fit(self.data.values)
        labels = kms.labels_
        print(len(labels),len(self.data.index))
        # self.data['Cluster'] = labels
        # cluster_distrib = self.data['Cluster'].value_counts()
        # sns.barplot(x=cluster_distrib.index, y=cluster_distrib.values, color='b')
        clusters = pd.concat(
            [self.data, pd.DataFrame({'cluster': labels})], axis=1)
        # print(clusters[['Country', 'clusters']])

        for c in clusters:
            grid = sns.FacetGrid(clusters, col='cluster')
            grid.map(plt.hist, c)
        # plt.show()
        print(kms.feature_importances_)

        pca = PCA(2)
        dist = 1 - cosine_similarity(self.X)
        pca.fit(dist)
        X_PCA = pca.transform(dist)
        colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow', 
          4: 'orange',  
          5:'purple'}

        names = {0: 'cluster 0', 
                1: 'cluster 1', 
                2: 'cluster 2', 
                3: 'cluster 3', 
                4: 'cluster 4',
                5:'cluster 5'}
        
        x, y = X_PCA[:, 0], X_PCA[:, 1]
  
       
        countries = []
        cnt = 0
        for cr in self.cntr:
            print(labels[cnt],cr)
            countries.append(cr)
            cnt+=1
        df = pd.DataFrame({'x': x, 'y':y, 'label':labels,'Country':countries}) 
        groups = df.groupby('label')

        

        fig, ax = plt.subplots(figsize=(20, 13)) 

        x = []
        y = []
        for name, group in groups:
            x.append(group.x)
            y.append(group.y)
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=5,
                    color=colors[name],label=names[name], mec='none')
            for i in range(len(group['x'].values)):
                ax.text(group['x'].values[i],group['y'].values[i],group['Country'].values[i])
            ax.set_aspect('auto')
            ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
            
        # print(x,y)

        ax.legend()
        ax.set_title("Country Segmentation based on primary energy consumption.")
        plt.show()


if __name__ == '__main__':
    data = Data("Energy-Data-Edited.csv", 2022)
    data.pre_process()
    print(data.list_features())
    columns = ['Country', 'co2_mtco2', 'elect_twh',
               'oilcons_kbd', 'coalcons_ej', 'primary_ej']
    # for cols in columns[1:]:
    #     data.remove_outliers(cols)
        # print(data.list_features())
    model = Model(6, data)
    # model.elbow()
    model.fit()
