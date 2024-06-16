import json
import osst.libosst as osst  # Import the OSST extension
from osst.model.tree_survival_regressor import TreeSurvivalRegressor  # Import the tree survival regressor model

class OSST:
    def __init__(self, configuration={}):
        self.configuration = configuration
        self.time = 0.0
        # self.stime = 0.0
        # self.utime = 0.0
        # self.maxmem = 0
        # self.numswap = 0
        # self.numctxtswitch = 0
        self.iterations = 0
        self.size = 0
        self.tree = None
        self.encoder = None
        self.lb = 0
        self.ub = 0
        self.timeout = False
        self.reported_loss = 0

    def load(self, path):
        """
        Parameters
        ---
        path : string
            path to a JSON file representing a model
        """
        with open(path, 'r') as model_source:
            result = model_source.read()
        result = json.loads(result)
        self.tree = TreeSurvivalRegressor(result[0])

    def __train__(self, X, event, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            matrix containing the training samples and features
        event: array-like, shape = [n_samples by 1]
            an n-by-1 column of event associated with each sample
        y : array-like, shape = [n_samples by 1]
            column containing the correct label for each sample in X
        Modifies
        ---
        trains a model using the OSST native extension
        """
        (n, m) = X.shape
        dataset = X.copy()
        dataset.insert(m, "event", event)
        dataset.insert(m + 1, "time", y)  # It is expected that the last column is the label column

        osst.configure(json.dumps(self.configuration, separators=(',', ':')))
        result = osst.fit(dataset.to_csv(index=False))  # Perform extension call to train the model

        self.time = osst.time()  # Record the training time

        if osst.status() == 0:
            print("osst reported successful execution")
            self.timeout = False
        elif osst.status() == 2:
            print("osst reported possible timeout.")
            self.timeout = True
            self.time = -1
            # self.stime = -1
            # self.utime = -1
        else:
            print('----------------------------------------------')
            print(result)
            print('----------------------------------------------')
            raise Exception("Error: OSST encountered an error while training")

        result = json.loads(result)  # Deserialize resu

        self.tree = TreeSurvivalRegressor(result[0], X=X, event=event, y=y)  # Parse the first result into model
        self.iterations = osst.iterations()  # Record the number of iterations
        self.size = osst.size()  # Record the graph size required

        # self.maxmem = osst.maxmem()
        # self.numswap = osst.numswap()
        # self.numctxtswitch = osst.numctxtswitch()

        self.lb = osst.lower_bound()  # Record reported global lower bound of algorithm
        self.ub = osst.upper_bound()  # Record reported global upper bound of algorithm
        self.reported_loss = osst.model_loss()  # Record reported training loss of returned tree

        print("training completed. {:.3f} seconds.".format(self.time))
        print("bounds: [{:.6f}..{:.6f}] ({:.6f}) IBS loss = {:.6f}, iterations={}".format(self.lb, self.ub, self.ub - self.lb,
                                                                                    self.reported_loss,
                                                                                    self.iterations))

    def fit(self, X, event, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            matrix containing the training samples and features
        event: array-like, shape = [n_samples by 1]
            an n-by-1 column of event associated with each sample
        y : array-like, shape = [n_samples by 1]
            column containing the correct label for each sample in X
        Modifies
        ---
        trains the model so that this model instance is ready for prediction
        """

        self.__train__(X, event, y)

        return self

    def predict(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        Returns
        ---
        array-like, shape = [n_sampels by 1] : a column where each element is the leaf node index associated with each row
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.predict(X)

    def score(self, X, event, y):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            an n-by-m matrix of sample and their features
        event: array-like, shape = [n_samples by 1]
            an n-by-1 column of event associated with each sample
        y : array-like, shape = [n_samples by 1]
            an n-by-1 column of labels associated with each sample
            
        Returns
        ---
        real number : IBS score
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.score(X, event, y)

    def __len__(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return len(self.tree)

    def leaves(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.leaves()

    def nodes(self):
        """
        Returns
        ---
        natural number : The number of nodes present in this tree
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.nodes()

    def max_depth(self):
        """
        Returns
        ---
        natural number : the length of the longest decision path in this tree. A single-node tree will return 1.
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.maximum_depth()

    def latex(self):
        """
        Note
        ---
        This method doesn't work well for label headers that contain underscores due to underscore being a reserved character in LaTeX
        Returns
        ---
        string : A LaTeX string representing the model
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.latex()

    def json(self):
        """
        Returns
        ---
        string : A JSON string representing the model
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.json()

    def predict_survival_function(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        Returns
        ---
        survival functions: array-like, shape = [n_sampels, ]
             a column where each element is the predicted survival function associated with each row
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.predict_survival_function(X)
    
    def predict_cumulative_harzard_function(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        Returns
        ---
        cumulative hazard function: array-like, shape = [n_sampels, ]
            a column where each element is the predicted cumulative harzard function associated with each row
        """
        if self.tree is None:
            raise Exception("Error: Model not yet trained")
        return self.tree.predict_cumulative_harzard_function(X)