import numpy, theano
from theano import tensor as T

class PreferenceModel(object):
  def __init__(self, pref_type, modelname, hidden_size, ont_rep, vocab_rep=None, lr_wp_rank = 10):
    """
    pref_type (string):  Indicate whether the preference being modeled is word-concept or concept-concept (word_concept, concept_concept)
    modelname (string):  Select the parameterization of the model ('lincomb')
    ont_rep (theano.shared):  Shared ontology representation
    vocab_rep (theano.shared):  Shared vocabulary representation (required if pref_type is word-concept)
    """
    if pref_type == 'word_concept':
      self.pref_type = pref_type
      if vocab_rep is None:
        raise RuntimeError, "vocab_rep has to be given when pref_type is word_concept"
    else:
      self.pref_type = 'concept_concept'
    impl_models = ['linlayer', 'tanhlayer', 'weighted_prod', 'lr_weighted_prod']
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

    if self.modelname == 'linlayer' or self.modelname == 'tanhlayer':
      proj_weight_dim = ent1_dim + ent2_dim
      proj_weight_range = numpy.sqrt(6. / (proj_weight_dim + hidden_size))
      init_proj_weight = numpy.asarray(numpy_rng.uniform(low = -proj_weight_range, high = proj_weight_range, size = (proj_weight_dim, hidden_size)))
      self.proj_weight = theano.shared(value=init_proj_weight, name='P_p')
      score_weight_range = numpy.sqrt(6. / hidden_size)
      init_score_weight = numpy.asarray(numpy_rng.uniform(low = -score_weight_range, high = score_weight_range, size = hidden_size))
      self.score_weight = theano.shared(value=init_score_weight, name='p_s')
      self.params = [self.proj_weight, self.score_weight]
    elif self.modelname == 'weighted_prod':
      init_prod_weight = numpy.asarray(numpy_rng.uniform(low = -1.0, high = 1.0, size = (ent1_dim, ent2_dim)))
      self.prod_weight = theano.shared(value = init_prod_weight, name = 'P_w')
      self.params = [self.prod_weight]
    elif self.modelname == 'lr_weighted_prod':
      init_prod_weight1 = numpy.asarray(numpy_rng.uniform(low = -1.0, high = 1.0, size = (ent1_dim, lr_wp_rank)))
      init_prod_weight2 = numpy.asarray(numpy_rng.uniform(low = -1.0, high = 1.0, size = (lr_wp_rank, ent2_dim)))
      self.prod_weight1 = theano.shared(value = init_prod_weight1, name = 'P_w1')
      self.prod_weight2 = theano.shared(value = init_prod_weight2, name = 'P_w2')
      self.params = [self.prod_weight1, self.prod_weight2]

  def get_symb_score(self, ent1_ind, ent2_ind):
    if self.pref_type == 'word_concept':
      ent1_vec = self.vocab_rep[ent1_ind]
    else:
      ent1_vec = self.ont_rep[ent1_ind]
    ent2_vec = self.ont_rep[ent2_ind]
    if self.modelname == 'linlayer':
      pref_score = T.dot( self.score_weight, T.dot(self.proj_weight.T, T.concatenate([ent1_vec, ent2_vec])))
    elif self.modelname == 'tanhlayer':
      pref_score = T.dot( self.score_weight, T.tanh(T.dot(self.proj_weight.T, T.concatenate([ent1_vec, ent2_vec])))) 
    elif self.modelname == 'weighted_prod':
      pref_score = T.dot(T.dot(ent1_vec, self.prod_weight), ent2_vec)
    elif self.modelname == 'lr_weighted_prod':
      pref_score = T.dot(T.dot(ent1_vec, self.prod_weight1), T.dot(self.prod_weight2, ent2_vec))
    return pref_score

  def get_params(self):
    return self.params

  def set_params(self, params):
    for i, param in enumerate(params):
      self.params[i].set_value(param)

