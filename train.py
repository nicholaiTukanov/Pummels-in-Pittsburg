import cleaning as clean
import sklearn as sk
import sklearn.model_selection as ms
import numpy as np
from sklearn.model_selection import GridSearchCV
#from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn import tree, svm, ensemble, neighbors, naive_bayes, neural_network
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import pickle

models_names = ['DecisionTree','KNN','SVM','RandomForest','MLP','AdaBoost','NaiveBayes']

def predict_severity(df, pipelines):
    # Remove unknown max severity levels
    df = clean.drop_rows_by_value(df, 'MAX_SEVERITY_LEVEL', [8,9])

    features, labels = df.drop(columns=['MAX_SEVERITY_LEVEL']), df.MAX_SEVERITY_LEVEL

    for i,pipeline in enumerate(pipelines):
        grid_searcher = ms.GridSearchCV(pipeline[2], pipeline[1], scoring='accuracy', cv=5, n_jobs=-1)
        grid_searcher.fit(features, labels)

        # Save the model 
        model = grid_searcher.best_estimator_
        filename = '%s.sav'%models_names[i]
        pickle.dump(model, open(filename, 'wb'))

        predictions = ms.cross_val_predict(grid_searcher, features, y=labels, cv=10, verbose=1, n_jobs=-1)

        print("\nInformation for " + pipeline[0] + " cross-validated model")
        print("\nAccuracy = {:0.4f}".format(sk.metrics.accuracy_score(labels, predictions)))
        print("\nConfusion Matrix:")
        print(sk.metrics.confusion_matrix(labels, predictions))
        print("\nClassification Report")
        print(sk.metrics.classification_report(labels, predictions))

        if i == 3:
            display_rf_feature_importances(features, model)

def display_rf_feature_importances(features, model):
    def get_weight(item):
        return item[1]

    feature_weights = zip(features.columns, model.named_steps["classifier"].feature_importances_)
    feature_weights = sorted(feature_weights, key=get_weight)

    print("Most important features for Random Forest:")
    print([x[0] for x in feature_weights[:20]])

def get_models():
    models = []
    for model in models_names:
        filename = '%s.sav'%model
        loaded_model = pickle.load(open(filename, 'rb'))
        models.append(loaded_model)
    return models