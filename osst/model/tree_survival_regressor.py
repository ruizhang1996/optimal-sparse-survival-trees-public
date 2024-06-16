import pandas as pd
import numpy as np
from json import dumps, JSONEncoder
from numpy import array
from .utils import BaseEstimator
from .metrics import _integrated_brier_score
# Supporting Override for Converting Numpy Types into Python Values
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)


class TreeSurvivalRegressor:
    """
    Unified representation of a survival tree regressor in Python

    This class accepts a dictionary representation of a tree regressor and decodes it into an interactive object

    Additional support for encoding/decoding layer can be layers if the feature-space of the model differs from the feature space of the original data
    """

    def __init__(self, source, encoder=None, X=None, event=None, y=None):
        self.source = source  # The regressor stored in a recursive dictionary structure
        self.encoder = encoder  # Optional encoder / decoder unit to run before / after prediction

        
        if X is not None and event is not None and y is not None:  # Original training features and labels to fill in missing training loss values

            if len(event.shape) > 1:
                event = event.values.reshape(-1)
            if len(y.shape) > 1:
                y = y.values.reshape(-1)
            self.__initialize_training_loss__(X, event, y)
            # self.G = BaseEstimator(event=event, y=y, reverse=True)

    def __initialize_training_loss__(self, X, event, y):
        """
        Compares every prediction y_hat against the labels y, then incorporates the misprediction into the stored loss values
        This is used when parsing models from an algorithm that doesn't provide the training loss in the out put

        Parameters
        ---
        X : matrix-like, shape = [n_samples, m_features]
            matrix containing the training samples and features
        event: array-like, shape = [n_samples, ]
            an n-by-1 column of event associated with each sample
        y : array-like, shape = [n_samplee, ]
            column containing the correct label for each sample in X

        """
        self.G = BaseEstimator(event=event, y=y, reverse=True)
        (n, m) = X.shape
        predicted_leaf = self.predict(X)
        for node in self.__all_leaves__():
            node["loss"] = 0.0
            captured = (predicted_leaf == node['prediction'])
            # captured_X = X[captured]
            captured_event = event[captured]
            captured_y = y[captured]
            node["estimator"] = BaseEstimator(event=captured_event, y=captured_y)
            


        return

    def __find_leaf__(self, sample):
        """
        Returns
        ---
        the leaf by which this sample would be classified        
        """
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                return node
            else:
                value = sample[node["feature"]]
                reference = node["reference"]
                if node["relation"] == "==":
                    if value == reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == ">=":
                    if value >= reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == "<=":
                    if value <= reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == ">":
                    if value > reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == "<":
                    if value < reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                else:
                    raise "Unsupported relational operator {}".format(node["relation"])

    def __all_leaves__(self):
        """
        Returns
        ---
        list : a list of all leaves in this model
        """
        nodes = [self.source]
        leaf_list = []
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                leaf_list.append(node)
            else:
                nodes.append(node["true"])
                nodes.append(node["false"])
        return leaf_list

    def loss(self):
        """
        Returns
        ---
        real number : values between [0,1]
            the training loss of this model
        """
        return sum(node["loss"] for node in self.__all_leaves__())

    def classify(self, sample):
        """
        Parameters
        ---
        sample : array-like, shape = [m_features]
            a 1-by-m row representing each feature of a single sample

        Returns
        ---
        string : the prediction for a given sample and conditional probability (given the observations along the decision path) of it being correct
        """
        node = self.__find_leaf__(sample)
        return node["prediction"]

    def predict(self, X):
        """
        Requires
        ---
        the set of features used should be pre-encoding if an encoder is used

        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction

        Returns
        ---
        array-like, shape = [n_sampels by 1] : a column where each element is the prediction associated with each row
        """
        if self.encoder is not None:  # Perform an encoding if an encoding unit is specified
            X = pd.DataFrame(self.encoder.encode(X.values[:, :]), columns=self.encoder.headers)

        predictions = []
        (n, m) = X.shape
        for i in range(n):
            prediction = self.classify(X.values[i, :])
            predictions.append(prediction)
        return array(predictions)

    def predict_survival_function(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        Returns
        ---
        array-like, shape = [n_sampels by 1] : a column where each element is the predicted survival function associated with each row
        """
        if self.encoder is not None:  # Perform an encoding if an encoding unit is specified
            X = pd.DataFrame(self.encoder.encode(X.values[:, :]), columns=self.encoder.headers)

        predictions = []
        (n, m) = X.shape
        for i in range(n):
            leaf_node = self.__find_leaf__(X.values[i, :])
            predictions.append(leaf_node["estimator"].predict_survival_prob)
        return array(predictions)

    def predict_cumulative_harzard_function(self, X):
        """
        Parameters
        ---
        X : matrix-like, shape = [n_samples by m_features]
            a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction
        Returns
        ---
        array-like, shape = [n_sampels by 1] : a column where each element is the predicted cumulative harzard function associated with each row
        """
        if self.encoder is not None:  # Perform an encoding if an encoding unit is specified
            X = pd.DataFrame(self.encoder.encode(X.values[:, :]), columns=self.encoder.headers)

        predictions = []
        (n, m) = X.shape
        for i in range(n):
            leaf_node = self.__find_leaf__(X.values[i, :])
            predictions.append(leaf_node["estimator"].predict_cumu_harzard_prob)
        return array(predictions)
    

    def score(self, X, event, y):
        """
        Parameters
        --- 
        X : matrix-like, shape = [n_samples by m_features]
            an n-by-m matrix of sample and their features
        event: array-like, shape = [n_samples by 1] or [n_samples,]
            an n-by-1 column of event associated with each sample
        y : array-like, shape = [n_samples by 1] or [n_samples,]
            an n-by-1 column of labels associated with each sample
        Returns
        ---
        real number : the accuracy produced by applying this model overthe given dataset, with optionals for weighted accuracy
        """
        survival_functions = self.predict_survival_function(X)

        if len(event.shape) > 1:
            event = event.values.reshape(-1)
        if len(y.shape) > 1:
            y = y.values.reshape(-1)

        times = np.unique(y) 
        estimates = np.array([f(times) for f in survival_functions])
    
        return _integrated_brier_score(self.G, event, y, estimates, times)

    def __len__(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        return self.leaves()

    def leaves(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        leaves_counter = 0
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                leaves_counter += 1
            else:
                nodes.append(node["true"])
                nodes.append(node["false"])
        return leaves_counter

    def nodes(self):
        """
        Returns
        ---
        natural number : The number of nodes present in this tree
        """
        nodes_counter = 0
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                nodes_counter += 1
            else:
                nodes_counter += 1
                nodes.append(node["true"])
                nodes.append(node["false"])
        return nodes_counter

    def features(self):
        """
        Returns
        ---
        set : A set of strings each describing the features used by this model
        """
        feature_set = set()
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                continue
            else:
                feature_set.add(node["name"])
                nodes.append(node["true"])
                nodes.append(node["false"])
        return feature_set

    def encoded_features(self):
        """
        Returns
        ---
        natural number : The number of encoded features used by the supplied encoder to represent the data set
        """
        return len(self.encoder.headers) if self.encoder is not None else None

    def maximum_depth(self, node=None):
        """
        Returns
        ---
        natural number : the length of the longest decision path in this tree. A single-node tree will return 1.
        """
        if node is None:
            node = self.source
        if "prediction" in node:
            return 1
        else:
            return 1 + max(self.maximum_depth(node["true"]), self.maximum_depth(node["false"]))

    def __str__(self):
        """
        Returns
        ---
        string : pseuodocode representing the logic of this regressor
        """
        cases = []
        for group in self.__groups__():
            predicates = []
            for name in sorted(group["rules"].keys()):
                domain = group["rules"][name]
                if domain["type"] == "Categorical":
                    if len(domain["positive"]) > 0:
                        predicates.append("{} = {}".format(name, list(domain["positive"])[0]))
                    elif len(domain["negative"]) > 0:
                        if len(domain["negative"]) > 1:
                            predicates.append(
                                "{} not in {{ {} }}".format(name, ", ".join([str(v) for v in domain["negative"]])))
                        else:
                            predicates.append("{} != {}".format(name, str(list(domain["negative"])[0])))
                    else:
                        raise "Invalid Rule"
                elif domain["type"] == "Numerical":
                    predicate = name
                    if domain["min"] != -float("INF"):
                        predicate = "{} <= ".format(domain["min"]) + predicate
                    if domain["max"] != float("INF"):
                        predicate = predicate + " < {}".format(domain["max"])
                    predicates.append(predicate)

            if len(predicates) == 0:
                condition = "if true then:"
            else:
                condition = "if {} then:".format(" and ".join(predicates))
            outcomes = []
            # for outcome, probability in group["distribution"].items():
            outcomes.append("    predicted {}: {}".format(group["name"], group["prediction"]))
            outcomes.append("    normalized loss penalty: {}".format(round(group["loss"], 3)))
            outcomes.append("    complexity penalty: {}".format(round(group["complexity"], 3)))
            result = "\n".join(outcomes)
            cases.append("{}\n{}".format(condition, result))
        return "\n\nelse ".join(cases)

    def __repr__(self):
        """
        Returns
        ---
        dictionary : The recursive dictionary used to represent the model
        """
        return self.source

    def latex(self, node=None):
        """
        Note
        ---
        This method doesn't work well for label headers that contain underscores due to underscore being a reserved character in LaTeX

        Returns
        ---
        string : A LaTeX string representing the model
        """
        if node is None:
            node = self.source
        if "prediction" in node:
            if "name" in node:
                name = node["name"]
            else:
                name = "feature_{}".format(node["feature"])
            return "[ ${}$ [ ${}$ ] ]".format(name, node["prediction"])
        else:
            if "name" in node:
                if "=" in node["name"]:
                    name = "{}".format(node["name"])
                else:
                    name = "{} {} {}".format(node["name"], node["relation"], node["reference"])
            else:
                name = "feature_{} {} {}".format(node["feature"], node["relation"], node["reference"])
            return "[ ${}$ {} {} ]".format(name, self.latex(node["true"]), self.latex(node["false"])).replace("==",
                                                                                                              " \eq ").replace(
                ">=", " \ge ").replace("<=", " \le ")

    def json(self):
        """
        Returns
        ---
        string : A JSON string representing the model
        """
        return dumps(self.source, cls=NumpyEncoder)

    def __groups__(self, node=None):
        """
        Parameters
        --- 
        node : node within the tree from which to start
        Returns
        ---
        list : Object representation of each leaf for conversion to a case in an if-then-else statement
        """
        if node is None:
            node = self.source
        if "prediction" in node:
            node["rules"] = {}
            groups = [node]
            return groups
        else:
            if "name" in node:
                name = node["name"]
            else:
                name = "feature_{}".format(node["feature"])
            reference = node["reference"]
            groups = []
            for condition_result in ["true", "false"]:
                subtree = node[condition_result]
                for group in self.__groups__(subtree):

                    # For each group, add the corresponding rule
                    rules = group["rules"]
                    if not name in rules:
                        rules[name] = {}
                    rule = rules[name]
                    if node["relation"] == "==":
                        rule["type"] = "Categorical"
                        if "positive" not in rule:
                            rule["positive"] = set()
                        if "negative" not in rule:
                            rule["negative"] = set()
                        if condition_result == "true":
                            rule["positive"].add(reference)
                        elif condition_result == "false":
                            rule["negative"].add(reference)
                        else:
                            raise "OptimalSparseSurvivalTree: Malformatted source {}".format(node)
                    elif node["relation"] == ">=":
                        rule["type"] = "Numerical"
                        if "max" not in rule:
                            rule["max"] = float("INF")
                        if "min" not in rule:
                            rule["min"] = -float("INF")
                        if condition_result == "true":
                            rule["min"] = max(reference, rule["min"])
                        elif condition_result == "false":
                            rule["max"] = min(reference, rule["max"])
                        else:
                            raise "OptimalSparseSurvivalTree: Malformatted source {}".format(node)
                    else:
                        raise "Unsupported relational operator {}".format(node["relation"])

                    # Add the modified group to the group list
                    groups.append(group)
            return groups

    