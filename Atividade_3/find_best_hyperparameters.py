import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


class BestParam:
    def __init__(self, source_path: str = ".", target_path: str = ".") -> None:
        self.source_path = source_path
        self.target_path = target_path

    def __load_data(self):
        """Load data from the source CSV in source path."""
        return pd.read_csv(self.source_path)

    def __vectorize_text(self, texts: pd.Series):
        """Convert the text data into a sparse matrix using CountVectorizer."""
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(texts), vectorizer

    def __save_data(self, data, filename: str):
        """Save the data to the target path as a CSV file."""
        pd.DataFrame(data.toarray() if hasattr(data, 'toarray') else data).to_csv(
            self.target_path + f'/{filename}', index=False
        )

    def __split_and_save_data(self, texts, classes, seed: int=-1):
        """Split the data into training and testing sets and save to CSV."""

        # If seed is -1, use a random seed
        if seed == -1:
            seed = np.random.randint(0, 2**32 - 1)  # Generate a random seed
        
        # Split the data into train and test
        text_train, text_test, class_train, class_test = train_test_split(
            texts, classes, test_size=0.2, random_state=seed
        )
        
        # Save the data to CSV
        self.__save_data(text_train, 'text_train.csv')
        self.__save_data(class_train, 'class_train.csv')
        self.__save_data(text_test, 'text_test.csv')
        self.__save_data(class_test, 'class_test.csv')


    def prepare_data(self, seed: int):
        """Load csv, vectorize, split, and save the data into target path."""
        data = self.__load_data()
        texts = data["text"]
        classes = data["class"]

        # Vectorize the texts
        texts_matrix, vectorizer = self.__vectorize_text(texts)

        # Split and save the data
        self.__split_and_save_data(texts_matrix, classes, seed)

    def __prepare_model_search(self):
        """Prepare the models and their parameter grids for the grid search."""
        return [
            ('MultinomialNB', MultinomialNB(), 
             {'alpha': np.logspace(-3, 3, 7), 'fit_prior': [True, False]}),
            ('LogisticRegression', LogisticRegression(), 
             {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['lbfgs', 'liblinear', 'saga']}),
            ('SVC', SVC(), {'C': [0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf', 'poly']})
        ]

    def __run_grid_search(self, model, param_grid, texts, classes, cv):
        """Run grid search for a given model and parameters."""
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=1)
        grid_search.fit(texts, classes)
        return grid_search

    def __collect_grid_search_results(self, grid_search, model_name):
        """Collect the results from the grid search."""
        results = []
        for params, mean_score in zip(grid_search.cv_results_["params"], grid_search.cv_results_["mean_test_score"]):
            results.append({
                "Model": model_name,
                "Params": params,
                "Mean Test Score": mean_score,
            })
        return results

    def grid_search(self, cv):
        """Perform grid search on multiple models and save the results on target path."""
        # Load the training data
        texts = csr_matrix(pd.read_csv(self.target_path + '/text_train.csv'))
        classes = pd.read_csv(self.target_path + '/class_train.csv').squeeze()

        # Prepare models and parameters
        models = self.__prepare_model_search()

        # Store results for all models
        all_results = []
        for name, model, param_grid in models:
            # Run grid search
            grid_search = self.__run_grid_search(model, param_grid, texts, classes, cv)
            
            # Collect results
            model_results = self.__collect_grid_search_results(grid_search, name)
            all_results.extend(model_results)

        # Sort the results
        all_results.sort(key=lambda x: (x["Model"], -x["Mean Test Score"]))

        # Save the results to a CSV file
        pd.DataFrame(all_results).to_csv(self.target_path + '/model_search_results.csv', index=False)
