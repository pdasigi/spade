import numpy, theano
from theano import tensor as T

class HypernymModel(object):
	def __init__(self, modelname, hidden_size, vocab_rep, ont_rep):
		"""
		modelname (string):	Select the parameterization of the model ('lincomb')
		vocab_rep (theano.shared):	Shared vocabulary representation
		ont_rep (theano.shared):	Shared ontology representation
		"""
		impl_models = ['linlayer']
		if modelname not in impl_models:
			raise NotImplementedError, "Model name %s not known"%(modelname)
		numpy_rng = numpy.random.RandomState(12345)
		self.modelname = modelname
		self.vocab_rep = vocab_rep
		self.ont_rep = ont_rep
		vocab_size, word_dim = vocab_rep.get_value().shape
		ont_size, concept_dim = ont_rep.get_value().shape
		if self.modelname == 'linlayer':
			proj_weight_dim = word_dim + concept_dim
			proj_weight_range = 4 * numpy.sqrt(6. / (proj_weight_dim + hidden_size))
			init_proj_weight = numpy.asarray(numpy_rng.uniform(low = -proj_weight_range, high = proj_weight_range, size = (proj_weight_dim, hidden_size)))
			self.proj_weight = theano.shared(value=init_proj_weight, name='H_p')
			score_weight_range = 4 * numpy.sqrt(6. / hidden_size)
			init_score_weight = numpy.asarray(numpy_rng.uniform(low = -score_weight_range, high = score_weight_range, size = hidden_size))
			self.score_weight = theano.shared(value=init_score_weight, name='h_s')
			self.params = [self.proj_weight, self.score_weight]

	def get_symb_score(self, word_ind, concept_ind):
		if self.modelname == 'linlayer':
			word_vec = self.vocab_rep[word_ind]
			concept_vec = self.ont_rep[concept_ind]
			self.score = T.dot( self.score_weight, T.dot( self.proj_weight.T,  T.concatenate([word_vec, concept_vec]) ) )
			return self.score
		else:
			raise NotImplementedError

	def get_params(self):
		return self.params
