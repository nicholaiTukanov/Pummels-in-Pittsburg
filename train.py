import cleaning as clean
import sklearn as sk
import sklearn.model_selection as ms
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn import tree, svm, ensemble, neighbors, naive_bayes, neural_network
import pickle

models_names= ['DecisionTree','KNN','SVM','RandomForest','MLP','AdaBoost','NaiveBayes']
def predict_severity(df):

    numeric_transformer = Pipeline(steps=[
        ("imputer", Imputer(missing_values='NaN', strategy="median", axis=0))])

    pipelines = [
        (
            "Decision Tree",
            {
                'classifier__max_depth' : [5,10,15,20], 
                'classifier__min_samples_leaf' : [5,10,15,20],
                'classifier__max_features' : [5,10,15]
            },
            Pipeline(steps=[('preprocessor', numeric_transformer),
                            ('classifier', tree.DecisionTreeClassifier(criterion='entropy'))])
        ),
        (
            "K Nearest Neighbor",
            {
                'classifier__n_neighbors' : [3, 5, 7], 
                'pca__n_components' : [10, 25, 50]
            },
            Pipeline(steps=[('preprocessor', numeric_transformer),
                            ('std_scaler', StandardScaler()),
                            ('pca', PCA()),
                            ('classifier', neighbors.KNeighborsClassifier())])
        ),
        (
            "Support Vector Machine",
            {
                'classifier__kernel' : ['linear', 'poly'], 
                'classifier__degree' : [3, 5],
                'pca__n_components' : [10, 25, 50]
            },
            Pipeline(steps=[('preprocessor', numeric_transformer),
                            ('std_scaler', StandardScaler()),
                            ('pca', PCA()),
                            ('classifier', svm.SVC())])
        ),
        (
            "Random Forest",
            {
                'classifier__max_depth' : [5,10,15,20], 
                'classifier__min_samples_leaf' : [5,10,15,20],
                'classifier__max_features' : [5,10,15],
                'classifier__n_estimators' : [10, 100, 1000]
            },
            Pipeline(steps=[('preprocessor', numeric_transformer),
                            ('classifier', ensemble.RandomForestClassifier())])
        ),
        (
            "Neural Net (MLP)",
            {
                'classifier__activation' : ['logistic', 'tanh', 'relu'],
                'classifier__hidden_layer_size' : list(range(30, 61, 10)) 
            },
            Pipeline(steps=[('preprocessor', numeric_transformer),
                            ('std_scaler', StandardScaler()),
                            ('classifier', neural_network.MLPClassifier())])
        ),
        (
            "AdaBoost",
            {
                'classifier__n_estimators' : [50, 100, 150],
            },
            Pipeline(steps=[('preprocessor', numeric_transformer),
                            ('classifier', ensemble.AdaBoostClassifier())])
        ),
        (
            "Gaussian Naive Bayes",
            {},
            Pipeline(steps=[('preprocessor', numeric_transformer),
                            ('classifier', naive_bayes.GaussianNB())])
        )
    ]

    features, labels = df.drop(
        columns=['MAX_SEVERITY_LEVEL']), df.MAX_SEVERITY_LEVEL

    for i,pipeline in enumerate(pipelines):
        grid_searcher = ms.GridSearchCV(pipeline[2], pipeline[1], scoring='accuracy', cv=5, n_jobs=-1)
        grid_searcher.fit(features, labels)
        model = grid_searcher.best_estimator_
        filename = '%s.sav'%models_names[i]
        pickle.dump(model, open(filename, 'wb'))
        scores = ms.cross_val_score(grid_searcher, features, y=labels, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
        print("Accuracy of " + pipeline[0] + " cross-validated model = {:0.4f}".format(sum(scores)/len(scores)))


    # # TODO: Use imputes from training data instead of test data
    # features_test = numeric_transformer.fit_transform(features_test)

    # prediction = clf.predict(features_test)
    # acc = sk.metrics.accuracy_score(prediction, labels_test)
    # print('Accuracy', acc)


def main():
    df = clean.get_clean_data()

    #Temporary: Only take 0.5% of the data while testing
    df = (df.head(int(len(df)*0.002)))

    #clean.data_info(df)
    predict_severity(df)

if __name__ == '__main__':
    main()
else:
    models =[]
    for model in models_names:
        filename = '%s.sav'%model
        loaded_model = pickle.load(open(filename, 'rb'))
        models.append(loaded_model)