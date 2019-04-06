import luigi
import pickle
import pandas as pd
from pathlib import Path

from feature_builder import FeatureBuilder
from booster import train

data_path = Path("data")
output_path = Path("output")

if not output_path.exists():
    output_path.mkdir()


class TrainDataIngestion(luigi.Task):

    def run(self):
        df_data = pd.read_csv(data_path.joinpath("train.csv"))
        df_data.to_csv(self.output().path, index=False)

    def output(self):
        return luigi.LocalTarget("output/titanic_train.csv")


class DataPreProcessing(luigi.Task):

    def requires(self):
        return TrainDataIngestion()

    def run(self):
        fb = FeatureBuilder(
            pd.read_csv(TrainDataIngestion().output().path))
        dataframe = fb.featurize()
        print("In Data Pre Processing")
        print(dataframe.columns)
        dataframe.to_csv(self.output().path, index=False)

    def output(self):
        return luigi.LocalTarget("output/titanic_train_clean.csv")


class Train(luigi.Task):

    def requires(self):
        return DataPreProcessing()

    def run(self):
        model = train(
            pd.read_csv(DataPreProcessing().output().path))
        with open(self.output().path, 'wb') as f:
            pickle.dump(model, f)

    def output(self):
        return luigi.LocalTarget("output/titanic_model.pkl")


if __name__ == '__main__':
    luigi.run()
