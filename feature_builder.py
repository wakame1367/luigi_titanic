import pandas as pd


class FeatureBuilder:
    def __init__(self, dataframe, training=True):
        self.dataframe = dataframe
        self.continuous_features = ["CompetitionDistance", "Date"]
        self.categorical_features = ["StoreType"]
        self.target = ["Survived"]
        self.training = training

    def featurize(self):
        drop_columns = ["PassengerId", "Cabin", "Ticket"]
        print("Subsetting Dataframe")
        self.subset_dataframe()
        print("Feature Engineering")
        dataframe = self.feature_engineering(self.dataframe)
        self.handle_missing_value()
        self.dataframe.drop(drop_columns, axis=1, inplace=True)
        return dataframe

    def feature_engineering(self, dataframe=None):
        pass
        return dataframe

    def subset_dataframe(self):
        if self.training:
            self.dataframe = self.dataframe[
                self.continuous_features + self.categorical_features + self.target]
        else:
            self.dataframe = self.dataframe[
                self.continuous_features + self.categorical_features]

    def handle_missing_value(self):
        self.dataframe["Age"].fillna(self.dataframe["Age"].median(),
                                     inplace=True)
        self.dataframe["Embarked"].fillna(self.dataframe["Embarked"].median(),
                                          inplace=True)
        self.dataframe["Fare"].fillna(self.dataframe["Fare"].median(),
                                      inplace=True)
