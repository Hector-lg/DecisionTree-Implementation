"""Base class for decision trees with a scikit-learn like API."""
from abc import ABC, abstractmethod
import numbers
import numpy as np
from .node import Node


class BaseDecisionTree(ABC):
    """Skeleton shared by every tree variant (classification, regression, ...).

    Hyperparameters are set in ``__init__``; ``fit``, ``predict`` and ``eval``
    follow the scikit-learn conventions. Subclasses implement the abstract
    hooks; the TODOs are left to the students.

    Parameters:
        max_depth (int or None): Maximum depth of the tree. The root has depth
            0. ``None`` means nodes grow until they are pure or smaller than
            ``min_samples_split``.
        min_samples_split (int): Minimum number of samples a node needs to be
            considered for splitting. Must be >= 2.
        criterion (str): Name of the impurity measure used to choose splits
            (e.g. 'gini', 'entropy', 'mse'). Valid values depend on the
            subclass.

    Attributes:
        root (Node or None): Root of the fitted tree; ``None`` before ``fit``.
    """

    def __init__(self, max_depth=None, min_samples_split=2, criterion=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def get_params(self):
        """Return the hyperparameters.

        Returns:
            dict: Maps 'max_depth', 'min_samples_split' and 'criterion' to
            their current values. Add new hyperparameters here.
        """
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "criterion": self.criterion,
        }

    def set_params(self, **params):
        """Update hyperparameters by keyword, e.g. ``set_params(max_depth=5)``.

        Args:
            **params: Hyperparameter names and new values.

        Returns:
            BaseDecisionTree: ``self``, to allow chaining.

        Raises:
            ValueError: If a name is not a known hyperparameter.
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """Build the tree from training data.

        TODO (students): implement this method. It should
          1. validate the hyperparameters (e.g. ``max_depth`` is None or a
             positive int, ``min_samples_split >= 2``, ``criterion`` is
             supported) and raise ``ValueError`` otherwise;
          2. convert ``X`` and ``y`` to arrays and check that they have the
             same number of samples;
          3. call ``self._build_tree(X, y, depth=0)`` and store the returned
             node in ``self.root``;
          4. return ``self``.

        Args:
            X (array-like, shape (n_samples, n_features)): Training features.
            y (array-like, shape (n_samples,)): Targets (class labels for
                classification, real values for regression).

        Returns:
            BaseDecisionTree: ``self`` (the fitted estimator).
        """
        if self.max_depth is not None:
            if (not isinstance(self.max_depth, numbers.Integral)
                    or isinstance(self.max_depth, bool)
                    or self.max_depth < 1):
                raise ValueError(
                    f"max_depth debe ser None o un entero positivo, "
                    f"se recibio {self.max_depth!r}"
                )
        if (not isinstance(self.min_samples_split, numbers.Integral)
                or isinstance(self.min_samples_split, bool)
                or self.min_samples_split < 2):
            raise ValueError(
                f"min_samples_split debe ser un entero >= 2, "
                f"se recibio {self.min_samples_split!r}"
            )
        criterios_validos = getattr(self, "_criterios_validos", None)
        if criterios_validos is not None and self.criterion not in criterios_validos:
            raise ValueError(
                f"criterion debe ser uno de {criterios_validos}, "
                f"se recibio {self.criterion!r}"
            )

        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2:
            raise ValueError(
                f"X debe ser 2D (n_samples, n_features), tiene forma {X.shape}"
            )
        if y.ndim != 1:
            raise ValueError(f"y debe ser 1D (n_samples,), tiene forma {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X y y deben tener el mismo numero de muestras: "
                f"{X.shape[0]} != {y.shape[0]}"
            )
        if X.shape[0] == 0:
            raise ValueError("No se puede entrenar con 0 muestras")

        self.root = self._build_tree(X, y, depth=0)

        return self


    def predict(self, X):
        """Predict a target for each row of ``X``.

        TODO (students): implement this method. It should raise an error if
        the model has not been fitted (``self.root is None``) and, for each
        sample, walk from ``self.root``: go left when
        ``x[feature_index] <= threshold`` else right, until a leaf is
        reached, then return that leaf's ``value``.

        Args:
            X (array-like, shape (n_samples, n_features)): Samples to predict.

        Returns:
            array, shape (n_samples,): Predicted target for every sample.
        """

        X = np.array(X)
        predicciones = []
        if self.root is None:
            raise ValueError("este arbol no ha sido fitteado")
        for i in range(X.shape[0]):
            root_copia = self.root
            while not root_copia.is_leaf:
                if X[i][root_copia.feature_index] <= root_copia.threshold:
                    root_copia = root_copia.left
                else:
                    root_copia = root_copia.right
                
            predicciones.append(root_copia.value)

        return np.array(predicciones)

        

    def eval(self, X, y):
        """Score the model on ``(X, y)``.

        TODO (students): implement this method. It should call
        ``self.predict(X)`` and return ``self._score(y, y_pred)``.

        Args:
            X (array-like, shape (n_samples, n_features)): Samples to evaluate.
            y (array-like, shape (n_samples,)): True targets for ``X``.

        Returns:
            float: Metric value (accuracy for classification, R^2 for
            regression; higher is better).
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X y y deben tener el mismo numero de muestras: "
                f"{X.shape[0]} != {y.shape[0]}"
            )
        y_pred = self.predict(X)
        return self._score(y, y_pred)


    @abstractmethod
    def _build_tree(self, X, y, depth):
        """Recursively grow the tree and return its root ``Node``.

        TODO (students): create a leaf (``Node(value=self._leaf_value(y))``)
        when a stopping condition holds: ``depth`` reached ``max_depth``,
        fewer than ``min_samples_split`` samples, ``y`` is pure, or
        ``_best_split`` finds no useful split. Otherwise split the data on the
        returned (feature, threshold) and recurse on both parts with
        ``depth + 1``.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Depth of this node (root is 0).

        Returns:
            Node: Subtree built from this data.
        """
        if depth == self.max_depth:
            return Node(value=self._leaf_value(y))
        if self.min_samples_split > X.shape[0]:
            return Node(value=self._leaf_value(y))
        

        split = self._best_split(X, y)

        if split is None:
            return Node(value=self._leaf_value(y))
        
        columna, umbral = split
        
        
        X_l = X[:, columna] <= umbral
        X_r = X[:, columna] > umbral

        X_left = X[X_l]
        X_right = X[X_r]
        y_l = y[X_l]
        y_r = y[X_r]

        left =self._build_tree(X_left, y_l, depth+1)
        right = self._build_tree(X_right, y_r, depth+1)

        return Node(columna, umbral, left, right )
        
    @abstractmethod
    def _impurity(self, y):
        """Compute the impurity of a set of targets.

        TODO (students): implement the measure selected by ``self.criterion``
        (e.g. Gini or entropy for classification, MSE or MAE for regression).
        Lower means more homogeneous; it is used to score candidate splits.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity of ``y`` (0 for a perfectly pure node).
        """
        unicos = list(set(y))
        gini = 0
        n = len(y)
        if n == 0:
            return 0
        for i in unicos:
            sumatoria = np.sum(y == i)/n
            sumatoria = sumatoria**2
            gini += sumatoria
        return 1 - gini


                
            

    @abstractmethod
    def _best_split(self, X, y):
        """Find the split that most reduces impurity.

        TODO (students): for each feature and candidate threshold (e.g.
        midpoints between sorted unique values), partition ``y`` into left
        (``x <= threshold``) and right, compute the weighted impurity
        ``(n_l * imp_l + n_r * imp_r) / n`` and keep the split with the
        largest gain over ``self._impurity(y)``. Skip splits leaving a side
        empty.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)`` of the
            best split, or ``None`` if no split improves impurity.
        """
        n = len(y)
        mejor_columna, mejor_umbral = None, None
        mejor_ganancia = 0
        for i in range(X.shape[1]):
            
            
            x_unique = np.unique(X[:,i])
            for j in range(len(x_unique)-1):
                umbral = (x_unique[j] + x_unique[j+1])/2
                y_l = X[:, i] <= umbral
                y_r = X[:, i] > umbral
                
                imp_l = self._impurity(y[y_l])
                imp_r = self._impurity(y[y_r])

                ganancia = self._impurity(y) - (len(y[y_l])/n * imp_l + len(y[y_r])/n * imp_r)
                

                if ganancia > mejor_ganancia:
                    mejor_ganancia = ganancia
                    mejor_columna = i
                    mejor_umbral = umbral

        if mejor_columna == None:
            return None
        
        return (mejor_columna, mejor_umbral)
            




    @abstractmethod
    def _leaf_value(self, y):
        """Compute the prediction stored in a leaf.

        TODO (students): return the majority class for classification or the
        mean of ``y`` for regression.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The single predicted value for the leaf.
        """

    @abstractmethod
    def _score(self, y_true, y_pred):
        """Compute the metric reported by ``eval``.

        TODO (students): e.g. accuracy for classification, R^2 for regression.

        Args:
            y_true (ndarray, shape (n_samples,)): Ground-truth targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        aciertos = 0
        
        n = len(y_true)
        for i in range(n):

            if y_pred[i] == y_true[i]:
                aciertos += 1

        return aciertos/n
