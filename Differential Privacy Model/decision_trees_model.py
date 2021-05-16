import numbers
import warnings
import pandas as pd
from abc import ABCMeta
from abc import abstractmethod
from math import ceil
import sklearn.tree as sk
import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.base import RegressorMixin
from sklearn.base import is_classifier
from sklearn.base import MultiOutputMixin
from sklearn.utils import Bunch
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.utils import compute_sample_weight
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import _deprecate_positional_args

from sklearn.tree._criterion import Criterion
from sklearn.tree._splitter import Splitter
from sklearn.tree._tree import DepthFirstTreeBuilder
from sklearn.tree._tree import BestFirstTreeBuilder
from sklearn.tree._tree import Tree
from sklearn.tree._tree import _build_pruned_tree_ccp
from sklearn.tree._tree import ccp_pruning_path
from sklearn.tree import _tree, _splitter, _criterion

from diffprivlib.mechanisms import laplace


__all__ = ["DecisionTreeClassifier",
           "DecisionTreeRegressor",
           "ExtraTreeClassifier",
           "ExtraTreeRegressor"]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {"gini": _criterion.Gini, "entropy": _criterion.Entropy}
CRITERIA_REG = {"mse": _criterion.MSE, "friedman_mse": _criterion.FriedmanMSE,
                "mae": _criterion.MAE}

DENSE_SPLITTERS = {"best": _splitter.BestSplitter,
                   "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {"best": _splitter.BestSparseSplitter,
                    "random": _splitter.RandomSparseSplitter}



#print(y_pred)
class DecisionTreeClassifier(sk.DecisionTreeClassifier):
	@_deprecate_positional_args
	def __init__(self, *,
	             criterion="gini",
	             splitter="best",
	             max_depth=None,
	             min_samples_split=2,
	             min_samples_leaf=1,
	             min_weight_fraction_leaf=0.,
	             max_features=None,
	             random_state=None,
	             max_leaf_nodes=None,
	             min_impurity_decrease=0.,
	             min_impurity_split=None,
	             class_weight=None,
	             presort='deprecated',
	             ccp_alpha=0.0, e, s):
		super().__init__(
			criterion=criterion,
			splitter=splitter,
			max_depth=max_depth,
			min_samples_split=min_samples_split,
			min_samples_leaf=min_samples_leaf,
			min_weight_fraction_leaf=min_weight_fraction_leaf,
			max_features=max_features,
			max_leaf_nodes=max_leaf_nodes,
			class_weight=class_weight,
			random_state=random_state,
			min_impurity_decrease=min_impurity_decrease,
			min_impurity_split=min_impurity_split,
			presort=presort,
			ccp_alpha=ccp_alpha)
		self.e = e
		self.s = s

	def fit(self, X, y, sample_weight=None, check_input=True,
	        X_idx_sorted=None):

		random_state = check_random_state(self.random_state)

		if self.ccp_alpha < 0.0:
			raise ValueError("ccp_alpha must be greater than or equal to 0")

		if check_input:
			# Need to validate separately here.
			# We can't pass multi_ouput=True because that would allow y to be
			# csr.
			check_X_params = dict(dtype=DTYPE, accept_sparse="csc")
			check_y_params = dict(ensure_2d=False, dtype=None)
			X, y = self._validate_data(X, y,
			                           validate_separately=(check_X_params,
			                                                check_y_params))
			if issparse(X):
				X.sort_indices()

				if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
					raise ValueError("No support for np.int64 index based "
					                 "sparse matrices")

		# Determine output settings
		n_samples, self.n_features_ = X.shape
		is_classification = is_classifier(self)

		y = np.atleast_1d(y)
		expanded_class_weight = None

		if y.ndim == 1:
			# reshape is necessary to preserve the data contiguity against vs
			# [:, np.newaxis] that does not.
			y = np.reshape(y, (-1, 1))

		self.n_outputs_ = y.shape[1]

		if is_classification:
			check_classification_targets(y)
			y = np.copy(y)
			#print(y)
			self.classes_ = []
			self.n_classes_ = []

			if self.class_weight is not None:
				y_original = np.copy(y)

			y_encoded = np.zeros(y.shape, dtype=np.int)
			for k in range(self.n_outputs_):
				classes_k, y_encoded[:, k] = np.unique(y[:, k],
				                                       return_inverse=True)
				self.classes_.append(classes_k)
				self.n_classes_.append(classes_k.shape[0])
			y = y_encoded

			if self.class_weight is not None:
				expanded_class_weight = compute_sample_weight(
					self.class_weight, y_original)

			self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)

		if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
			y = np.ascontiguousarray(y, dtype=DOUBLE)

		# Check parameters
		max_depth = (np.iinfo(np.int32).max if self.max_depth is None
		             else self.max_depth)
		max_leaf_nodes = (-1 if self.max_leaf_nodes is None
		                  else self.max_leaf_nodes)

		if isinstance(self.min_samples_leaf, numbers.Integral):
			if not 1 <= self.min_samples_leaf:
				raise ValueError("min_samples_leaf must be at least 1 "
				                 "or in (0, 0.5], got %s"
				                 % self.min_samples_leaf)
			min_samples_leaf = self.min_samples_leaf
		else:  # float
			if not 0. < self.min_samples_leaf <= 0.5:
				raise ValueError("min_samples_leaf must be at least 1 "
				                 "or in (0, 0.5], got %s"
				                 % self.min_samples_leaf)
			min_samples_leaf = int(ceil(self.min_samples_leaf * n_samples))

		if isinstance(self.min_samples_split, numbers.Integral):
			if not 2 <= self.min_samples_split:
				raise ValueError("min_samples_split must be an integer "
				                 "greater than 1 or a float in (0.0, 1.0]; "
				                 "got the integer %s"
				                 % self.min_samples_split)
			min_samples_split = self.min_samples_split
		else:  # float
			if not 0. < self.min_samples_split <= 1.:
				raise ValueError("min_samples_split must be an integer "
				                 "greater than 1 or a float in (0.0, 1.0]; "
				                 "got the float %s"
				                 % self.min_samples_split)
			min_samples_split = int(ceil(self.min_samples_split * n_samples))
			min_samples_split = max(2, min_samples_split)

		min_samples_split = max(min_samples_split, 2 * min_samples_leaf)

		if isinstance(self.max_features, str):
			if self.max_features == "auto":
				if is_classification:
					max_features = max(1, int(np.sqrt(self.n_features_)))
				else:
					max_features = self.n_features_
			elif self.max_features == "sqrt":
				max_features = max(1, int(np.sqrt(self.n_features_)))
			elif self.max_features == "log2":
				max_features = max(1, int(np.log2(self.n_features_)))
			else:
				raise ValueError("Invalid value for max_features. "
				                 "Allowed string values are 'auto', "
				                 "'sqrt' or 'log2'.")
		elif self.max_features is None:
			max_features = self.n_features_
		elif isinstance(self.max_features, numbers.Integral):
			max_features = self.max_features
		else:  # float
			if self.max_features > 0.0:
				max_features = max(1,
				                   int(self.max_features * self.n_features_))
			else:
				max_features = 0

		self.max_features_ = max_features

		if len(y) != n_samples:
			raise ValueError("Number of labels=%d does not match "
			                 "number of samples=%d" % (len(y), n_samples))
		if not 0 <= self.min_weight_fraction_leaf <= 0.5:
			raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
		if max_depth <= 0:
			raise ValueError("max_depth must be greater than zero. ")
		if not (0 < max_features <= self.n_features_):
			raise ValueError("max_features must be in (0, n_features]")
		if not isinstance(max_leaf_nodes, numbers.Integral):
			raise ValueError("max_leaf_nodes must be integral number but was "
			                 "%r" % max_leaf_nodes)
		if -1 < max_leaf_nodes < 2:
			raise ValueError(("max_leaf_nodes {0} must be either None "
			                  "or larger than 1").format(max_leaf_nodes))

		if sample_weight is not None:
			sample_weight = _check_sample_weight(sample_weight, X, DOUBLE)

		if expanded_class_weight is not None:
			if sample_weight is not None:
				sample_weight = sample_weight * expanded_class_weight
			else:
				sample_weight = expanded_class_weight

		# Set min_weight_leaf from min_weight_fraction_leaf
		if sample_weight is None:
			min_weight_leaf = (self.min_weight_fraction_leaf *
			                   n_samples)
		else:
			min_weight_leaf = (self.min_weight_fraction_leaf *
			                   np.sum(sample_weight))

		min_impurity_split = self.min_impurity_split
		if min_impurity_split is not None:
			warnings.warn("The min_impurity_split parameter is deprecated. "
			              "Its default value has changed from 1e-7 to 0 in "
			              "version 0.23, and it will be removed in 0.25. "
			              "Use the min_impurity_decrease parameter instead.",
			              FutureWarning)

			if min_impurity_split < 0.:
				raise ValueError("min_impurity_split must be greater than "
				                 "or equal to 0")
		else:
			min_impurity_split = 0

		if self.min_impurity_decrease < 0.:
			raise ValueError("min_impurity_decrease must be greater than "
			                 "or equal to 0")

		if self.presort != 'deprecated':
			warnings.warn("The parameter 'presort' is deprecated and has no "
			              "effect. It will be removed in v0.24. You can "
			              "suppress this warning by not passing any value "
			              "to the 'presort' parameter.",
			              FutureWarning)

		# Build tree
		criterion = self.criterion
		if not isinstance(criterion, Criterion):
			if is_classification:
				criterion = CRITERIA_CLF[self.criterion](self.n_outputs_,
				                                         self.n_classes_)
			else:
				criterion = CRITERIA_REG[self.criterion](self.n_outputs_,
				                                         n_samples)

		SPLITTERS = SPARSE_SPLITTERS if issparse(X) else DENSE_SPLITTERS

		splitter = self.splitter
		if not isinstance(self.splitter, Splitter):
			splitter = SPLITTERS[self.splitter](criterion,
			                                    self.max_features_,
			                                    min_samples_leaf,
			                                    min_weight_leaf,
			                                    random_state)

		if is_classifier(self):
			self.tree_ = Tree(self.n_features_,
			                  self.n_classes_, self.n_outputs_)

		else:
			self.tree_ = Tree(self.n_features_,
			                  # TODO: tree should't need this in this case
			                  np.array([1] * self.n_outputs_, dtype=np.intp),
			                  self.n_outputs_)

		# Use BestFirst if max_leaf_nodes given; use DepthFirst otherwise
		if max_leaf_nodes < 0:
			builder = DepthFirstTreeBuilder(splitter, min_samples_split,
			                                min_samples_leaf,
			                                min_weight_leaf,
			                                max_depth,
			                                self.min_impurity_decrease,
			                                min_impurity_split)
		else:
			builder = BestFirstTreeBuilder(splitter, min_samples_split,
			                               min_samples_leaf,
			                               min_weight_leaf,
			                               max_depth,
			                               max_leaf_nodes,
			                               self.min_impurity_decrease,
			                               min_impurity_split)

		builder.build(self.tree_, X, y, sample_weight, X_idx_sorted)
		#print(self.tree_.children_left.shape)
		if self.n_outputs_ == 1 and is_classifier(self):
			self.n_classes_ = self.n_classes_[0]
			self.classes_ = self.classes_[0]
		#print(self.tree_.weighted_n_node_samples)

		e = self.e
		#print(e)
		# for i in range(self.tree_.value.shape[0]):
		#
		# 	for j in range(self.tree_.value.shape[2]):
		#
		# 		self.e = e /((self.tree_.value[i][0][j] + max_depth))
		# 		#print(self.tree_.value[i][0][j])
		# 		self.tree_.value[i][0][j] = self.addNoise(self.tree_.value[i][0][j])
		# 		#print(self.tree_.value[i][0][j])

		#print(self.tree_.value[0][0])

		for i in range(self.tree_.value.shape[0]):
			fr = np.sum(self.tree_.value[i][0])
			self.e = e / (fr + max_depth)
			self.tree_.value[i][0] = self.addNoise(self.tree_.value[i][0])

		self._prune_tree()
		#print(self.tree_.value[0][0])
		return self

	def _validate_X_predict(self, X, check_input):
		"""Validate X whenever one tries to predict, apply, predict_proba"""
		if check_input:
			X = check_array(X, dtype=DTYPE, accept_sparse="csr")
			if issparse(X) and (X.indices.dtype != np.intc or
			                    X.indptr.dtype != np.intc):
				raise ValueError("No support for np.int64 index based "
				                 "sparse matrices")

		n_features = X.shape[1]
		if self.n_features_ != n_features:
			raise ValueError("Number of features of the model must "
			                 "match the input. Model n_features is %s and "
			                 "input n_features is %s "
			                 % (self.n_features_, n_features))
		#print(self.tree_.n_node_samples[self.tree_.children_left != -1])
		return X

	def predict(self, X, check_input=True):
		"""Predict class or regression value for X.

		For a classification model, the predicted class for each sample in X is
		returned. For a regression model, the predicted value based on X is
		returned.

		Parameters
		----------
		X : {array-like, sparse matrix} of shape (n_samples, n_features)
			The input samples. Internally, it will be converted to
			``dtype=np.float32`` and if a sparse matrix is provided
			to a sparse ``csr_matrix``.

		check_input : bool, default=True
			Allow to bypass several input checking.
			Don't use this parameter unless you know what you do.

		Returns
		-------
		y : array-like of shape (n_samples,) or (n_samples, n_outputs)
			The predicted classes, or the predict values.
		"""
		check_is_fitted(self)
		X = self._validate_X_predict(X, check_input)
		proba = self.tree_.predict(X)

		#proba = self.addNoise(proba)
		#print(proba)
		n_samples = X.shape[0]


		# Classification
		if is_classifier(self):
			if self.n_outputs_ == 1:
				return self.classes_.take(np.argmax(proba, axis=1), axis=0)

			else:
				class_type = self.classes_[0].dtype
				predictions = np.zeros((n_samples, self.n_outputs_),
				                       dtype=class_type)
				for k in range(self.n_outputs_):
					predictions[:, k] = self.classes_[k].take(
						np.argmax(proba[:, k], axis=1),
						axis=0)

				return predictions


	def addNoise(self, value):
		#print(proba)
		lp = laplace.Laplace().set_epsilon(self.e).set_epsilon_delta(self.e, 0).set_sensitivity(1)
		noisy_counts = np.zeros(value.shape[0])
		for i in range(noisy_counts.shape[0]):
			noisy_counts[i] = lp.randomise(value[i])

		return noisy_counts
