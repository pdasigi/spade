import numpy, theano
from theano import tensor as T

class ReconstructionModel(object):
	def __init__(self, num_hyponym_list):
		"""
		num_hyponym_list (list(int)):	Number of hyponyms for concepts in the ontology
		"""
		self.srng = T.shared_randomstreams.RandomStreams(234)
		init_avgs = [x/2 for x in num_hyponym_list]
		init_stds = [numpy.sqrt(x/2) for x in num_hyponym_list]
		self.avgs = theano.shared(value=numpy.asarray(init_avgs), name='avg')
		self.stds = theano.shared(value=numpy.asarray(init_stds), name='std')

	def get_sym_rec_prob(self, concept_ind):
		#self.concept_ind = T.iscalar('c')
		avg, std = self.avgs[concept_ind], self.stds[concept_ind]
		self.r = self.srng.normal(size=(1,), avg=avg, std=std)[0] # 0 to convert the vector to a scalar
		self.p_r = 1./(std * numpy.sqrt(2 * numpy.pi )) * T.exp(- (self.r - avg) ** 2/(2 * std ** 2))	
		return (self.r, self.p_r)

	def get_param(self):
		return [self.avgs, self.stds]
