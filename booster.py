from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

target = "Survived"


def train(dataframe):
    model_object = {}
    model = RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(
        dataframe.drop(columns=[target], axis=1),
        dataframe[target])
    # model = model.fit(X_train, y_train)
    model.fit(X_train, y_train)
    model_object["model"] = model
    model_object["training_features"] = X_train.columns.tolist()
    return model_object
