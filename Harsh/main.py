import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns


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

        sorts = self.X[col].sort_values()
        Q1 = sorts.quantile(0.25)
        Q3 = sorts.quantile(0.75)

        IQR = Q3-Q1

        prev = len(self.X.index)

        self.X = sorts[~((sorts < (Q1 - 1.5 * IQR)) |
                         (sorts > (Q3 + 1.5 * IQR)))]

        print(f"{prev-len(self.X.index)} outliers were removed")

    def remove_all_outliers(self):
        Q1, Q3, IQR = self.IQR()
        prev = len(self.X.index)
        self.X = self.X[~((self.X < (Q1 - 1.5 * IQR)) |
                          (self.X > (Q3 + 1.5 * IQR))).any(axis=1)]
        print(f"{prev-len(self.X.index)} outliers were removed")


if __name__ == '__main__':
    data = Data("Energy-Data-Edited.csv", 2022)
    data.pre_process()
    # data.remove_all_outliers()
    # data.show_nulls()
    data.show_outliers('elect_twh')
    # print(data.IQR())
    # data.remove_outliers()
