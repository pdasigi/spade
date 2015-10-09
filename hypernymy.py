import numpy, theano
from theano import tensor as T

class HypernymModel(object):
  def __init__(self, modelname, hidden_size, vocab_rep, ont_rep):
    """
    modelname (string):  Select the parameterization of the model ('lincomb')
    vocab_rep (theano.shared):  Shared vocabulary representation
    ont_rep (theano.shared):  Shared ontology representation
    """
    impl_models = ['linlayer', 'tanhlayer', 'dotproduct', 'weighted_prod']
    if modelname not in impl_models:
      raise NotImplementedError, "Model name %s not known"%(modelname)
    numpy_rng = numpy.random.RandomState(12345)
    self.modelname = modelname
    self.vocab_rep = vocab_rep
    self.ont_rep = ont_rep
    vocab_size, word_dim = vocab_rep.get_value().shape
    ont_size, concept_dim = ont_rep.get_value().shape
    self.params = []
    if self.modelname == 'linlayer' or self.modelname == 'tanhlayer':
      proj_weight_dim = word_dim + concept_dim
      proj_weight_range = numpy.sqrt(6. / (proj_weight_dim + hidden_size))
      init_proj_weight = numpy.asarray(numpy_rng.uniform(low = -proj_weight_range, high = proj_weight_range, size = (proj_weight_dim, hidden_size)))
      self.proj_weight = theano.shared(value=init_proj_weight, name='H_p')
      score_weight_range = numpy.sqrt(6. / hidden_size)
      init_score_weight = numpy.asarray(numpy_rng.uniform(low = -score_weight_range, high = score_weight_range, size = hidden_size))
      self.score_weight = theano.shared(value=init_score_weight, name='h_s')
      self.params = [self.proj_weight, self.score_weight]
    elif self.modelname == 'weighted_prod':
      init_prod_weight = numpy.asarray(numpy_rng.uniform(low = -1.0, high = 1.0, size = (word_dim, concept_dim)))
      self.prod_weight = theano.shared(value = init_prod_weight, name = 'H_w')
      self.params = [self.prod_weight]

  def get_symb_score(self, word_ind, concept_ind):
    word_vec = self.vocab_rep[word_ind]
    concept_vec = self.ont_rep[concept_ind]
    if self.modelname == 'linlayer':
      self.score = T.dot( self.score_weight, T.dot( self.proj_weight.T,  T.concatenate([word_vec, concept_vec]) ) )
      return self.score
    elif self.modelname == 'tanhlayer':
      self.score = T.dot( self.score_weight, T.tanh(T.dot( self.proj_weight.T,  T.concatenate([word_vec, concept_vec]))))
      return self.score
    elif self.modelname == "dotproduct":
      self.score = T.dot(word_vec, concept_vec)
      return self.score
    elif self.modelname == 'weighted_prod':
      self.score = T.dot(T.dot(word_vec, self.prod_weight), concept_vec)
      return self.score
    else:
      raise NotImplementedError

  def get_params(self):
    return self.params

  def set_params(self, params):
    for i, param in enumerate(params):
      self.params[i].set_value(param)
