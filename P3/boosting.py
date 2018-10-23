import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		h=[]
		h = np.sum([np.array(self.clfs_picked[i].predict(features))*self.betas[i] for i in range(self.T)], axis=0)
		h = np.sign(h)
		return h.tolist()
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		w = np.ones(N)/N
		for t in range(self.T):
			e=100000000
			for clf in self.clfs:
				error = np.sum(w * np.not_equal(labels,clf.predict(features)))
				if error < e:
					h = clf
					e = error
					h_p = clf.predict(features)
			beta = np.log((1 - e) / e) / 2
			self.clfs_picked.append(h)
			self.betas.append(beta)
			for n in range(N):
				w[n] = w[n] * (np.exp(-beta) if labels[n] == h_p[n] else np.exp(beta))
			w = w / np.sum(w)

		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	