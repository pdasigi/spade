import numpy, theano
from theano import tensor as T

class PreferenceModel(object):
	def __init__(self, pref_type, modelname, hidden_size, ont_rep, vocab_rep=None):
		"""
		pref_type (string):	Indicate whether the preference being modeled is word-concept or concept-concept (word_concept, concept_concept)
		modelname (string):	Select the parameterization of the model ('lincomb')
		ont_rep (theano.shared):	Shared ontology representation
		vocab_rep (theano.shared):	Shared vocabulary representation (required if pref_type is word-concept)
		"""
		if pref_type == 'word_concept':
			self.pref_type = pref_type
			if vocab_rep is None:
				raise RuntimeError, "vocab_rep has to be given when pref_type is word_concept"
		else:
			self.pref_type = 'concept_concept'
		impl_models = ['linlayer']
		if modelname not in impl_models:
			raise NotImplementedError, "Model name %s not known"%(modelname)
		numpy_rng = numpy.random.RandomState(12345)
		self.modelname = modelname
		self.vocab_rep = vocab_rep
		self.ont_rep = ont_rep
		ont_size, concept_dim = ont_rep.get_value().shape
		if self.pref_type == 'word_concept':
			vocab_size, word_dim = vocab_rep.get_value().shape
			ent1_dim = word_dim
			ent2_dim = concept_dim
		else:
			ent1_dim = concept_dim
			ent2_dim = concept_dim

		if self.modelname == 'linlayer':
			proj_weight_dim = ent1_dim + ent2_dim
			proj_weight_range = 4 * numpy.sqrt(6. / (proj_weight_dim + hidden_size))
			init_proj_weight = numpy.asarray(numpy_rng.uniform(low = -proj_weight_range, high = proj_weight_range, size = (proj_weight_dim, hidden_size)))
			self.proj_weight = theano.shared(value=init_proj_weight, name='P_p')
			score_weight_range = 4 * numpy.sqrt(6. / hidden_size)
			init_score_weight = numpy.asarray(numpy_rng.uniform(low = -score_weight_range, high = score_weight_range, size = hidden_size))
			self.score_weight = theano.shared(value=init_score_weight, name='p_s')
			self.params = [self.proj_weight, self.score_weight]

	def get_symb_score(self, ent1_ind, ent2_ind):
		if self.pref_type == 'word_concept':
			ent1_vec = self.vocab_rep[ent1_ind]
		else:
			ent1_vec = self.ont_rep[ent1_ind]
		ent2_vec = self.ont_rep[ent2_ind]
		pref_score = T.dot( self.score_weight, T.dot(self.proj_weight.T, T.concatenate([ent1_vec, ent2_vec])))
		return pref_score

	def get_params(self):
		return self.params

