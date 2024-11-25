import pandas as pd
import numpy as np

from numpy import vectorize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


class BestParam:
    def __init__(self, source_path:str, target_path:str=".") -> None:
        self.source_path = source_path
        self.target_path = target_path

    def __load_data(self):
        return pd.read_csv(self.source_path)

    def split_data(self, seed: int):
        # Read csv file on source path
        data = self.__load_data()

        # Split the texts from their respectives classes
        texts = data["text"]
        classes = data["class"]
        
        # Transform the texts into a sparse table (word for each file)
        vectorizer = CountVectorizer()
        texts = vectorizer.fit_transform(texts)

        # Split the train/test 80%/20%
        text_train, text_test, class_train, class_test = train_test_split(texts, 
                                          classes, test_size=0.2,random_state=seed)


        # Save bag of words for training
        pd.DataFrame(text_train.toarray(), columns=
          vectorizer.get_feature_names_out()).to_csv(self.target_path + 
                        '/text_train.csv', index=False)
        
        # Save labels for training
        pd.DataFrame(class_train).to_csv(self.target_path + 
                        '/class_train.csv', index=False)
        
        # Save bag of words for testing
        pd.DataFrame(text_test.toarray(), columns=
          vectorizer.get_feature_names_out()).to_csv(self.target_path + 
                        '/text_test.csv', index=False)
        
        # Save labels for testing
        pd.DataFrame(class_test).to_csv(self.target_path + 
                        '/class_test.csv', index=False)

    def greedy_search(self, cv):
        # Read train data
        texts = csr_matrix(pd.read_csv(self.target_path + '/text_train.csv'))
        classes = pd.read_csv(self.target_path + '/class_train.csv').squeeze()

        # Models and parameters
        models = [('MultinomialNB', MultinomialNB(), 
                    {'alpha': np.logspace(-3, 3, 7),
                    'fit_prior': [True, False]}),

                  ('LogisticRegression', LogisticRegression(), 
                    {'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['lbfgs', 'liblinear', 'saga']}),

                  ('SVC', SVC(), {'C': [0.01, 0.1, 1, 10, 100],
                   'kernel': ['linear', 'rbf', 'poly']})
                ]

        res = []
        for name, model, param_grid in models:
            # Initiate grid search acording to model
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                                       cv=cv, n_jobs=-1, verbose=1)

            # Fit the data
            grid_search.fit(texts, classes)

            # Fill the results
            for params, mean_score in zip(grid_search.cv_results_["params"],
                                  grid_search.cv_results_["mean_test_score"]):
              res.append({
                "Model": name,
                "Params": params,
                "Mean Test Score": mean_score,
              })

        # Sort the results based on name then on test score
        res.sort(key=lambda x: (x["Model"], -x["Mean Test Score"]))

        # Save the Data Frame into a csv
        pd.DataFrame(res).to_csv(self.target_path + 
                        '/model_search_results.csv', index=False)
    