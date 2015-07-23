import numpy, theano
from theano import tensor as T

class ReconstructionModel(object):
	def __init__(self, ont_size, vocab_rep):
		"""
		ont_size (int):	Number of concepts in the ontology
		word_dim (int):	Dimensionality of word representations
		"""
		numpy_rng = numpy.random.RandomState(12345)
		_, word_dim = vocab_rep.get_value().shape
		self.vocab_rep = vocab_rep
		avg_range = 4 * numpy.sqrt(6. / (ont_size + word_dim))
		init_avgs = numpy.asarray(numpy_rng.uniform(low = -avg_range, high = avg_range, size = (ont_size, word_dim)))
		self.avgs = theano.shared(value = init_avgs, name = 'avgs')
		cov_range = 4 * numpy.sqrt(6. / (ont_size + word_dim + word_dim))
		init_covs = numpy.asarray(numpy_rng.uniform(low = -cov_range, high = cov_range, size = (ont_size, word_dim, word_dim)))	
		self.covs = theano.shared(value = init_covs, name = 'covs')

	def get_sym_rec_prob(self, word_ind, concept_ind):
		avg, cov = self.avgs[concept_ind], self.covs[concept_ind]
		word_rep = self.vocab_rep[word_ind]
		self.p_r = 1. / T.sqrt((2 * numpy.pi) ** 2 * T.nlinalg.det(cov)) * T.exp(- 0.5 * T.dot(T.dot((word_rep - avg), T.nlinalg.matrix_inverse(cov)), (word_rep - avg) ))
		return self.p_r

	def get_params(self):
		return [self.avgs, self.covs]
