import cleaning as clean
import sklearn as sk
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.preprocessing import Imputer


def predict_severity(df):
    features, labels = df.drop(
        columns=['MAX_SEVERITY_LEVEL']), df.MAX_SEVERITY_LEVEL

    features_train, features_test, labels_train, labels_test = \
        sk.model_selection.train_test_split(features, labels, test_size=0.2)

    numeric_transformer = Pipeline(steps=[
        ("imputer", Imputer(missing_values='NaN', strategy="median", axis=0))])

    clf = Pipeline(steps=[('preprocessor', numeric_transformer),
                          ('classifier', tree.DecisionTreeClassifier(criterion='entropy'))])

    clf.fit(features_train, labels_train)

    features_test = numeric_transformer.fit_transform(features_test)

    prediction = clf.predict(features_test)
    acc = sk.metrics.accuracy_score(prediction, labels_test)
    print('Accuracy', acc)


def main():
    df = clean.get_clean_data()
    clean.data_info(df)
    predict_severity(df)


main()
