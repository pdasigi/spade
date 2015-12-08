import numpy, theano
from theano import tensor as T

# TODO: Covariance parameters may be too simple. Think of a way to make the matrix positive semi-definite and non-singular

class ReconstructionModel(object):
  def __init__(self, ont_rep, vocab_rep, init_hyp_strengths=None, rec_model_type="gaussian"):
    """
    ont_rep:  Ontology rep theano shared variable
    vocab_rep:  Vocabulary rep theano shared variable
    init_hyp_strengths: A numpy matrix of size ont_size X vocab_size, with init_hyp_strengths[concept_ind][word_ind] showing the strength of word as a hypernym of concept (needed only for multinomial reconstruction)
    rec_model_type: multinomial or gaussian
    """
    if rec_model_type not in ["gaussian", "multinomial"]:
      raise NotImplementedError, "Unknown reconstruction model type: %s"%rec_model_type
    #numpy_rng = numpy.random.RandomState(12345)
    self.rec_model_type = rec_model_type
    if self.rec_model_type == "multinomial":
      self.hyp_strengths = theano.shared(value=init_hyp_strengths, name='hyp_strengths')
    _, word_dim = vocab_rep.get_value().shape
    self.vocab_rep = vocab_rep
    ont_size, _ = ont_rep.get_value().shape
    self.ont_rep = ont_rep
    #avg_range = 4 * numpy.sqrt(6. / (ont_size + word_dim))
    #init_avgs = numpy.asarray(numpy_rng.uniform(low = -avg_range, high = avg_range, size = (ont_size, word_dim)))
    #self.avgs = theano.shared(value = init_avgs, name = 'avgs')
    #init_cov_multiples = numpy.asarray([0.2]*ont_size)
    #self.cov_multiples = theano.shared(value = init_cov_multiples, name = 'cov_mult')
    self.word_dim = word_dim

  def get_sym_rec_prob(self, word_ind, concept_ind):
    #avg, cov_m = self.avgs[concept_ind], self.cov_multiples[concept_ind]
    if self.rec_model_type == "multinomial":
      # Softmax converts a vector to a matrix! Making it a vector with [0] before indexing
      self.p_r = (T.nnet.softmax(self.hyp_strengths[concept_ind])[0])[word_ind]
    else:
      avg = self.ont_rep[concept_ind]
      cov_m = 0.2
      word_rep = self.vocab_rep[word_ind]
      rep_m_avg = word_rep - avg
      exp_term = -0.5 * T.dot(rep_m_avg, rep_m_avg) * (1. / T.abs_(cov_m))
      sqrt_term = T.pow(2 * T.abs_(cov_m) * numpy.pi, self.word_dim)
      self.p_r = 1. / T.sqrt(sqrt_term) * T.exp(exp_term)
    return self.p_r

  def get_params(self):
    #return [self.avgs, self.cov_multiples]
    if self.rec_model_type == "multinomial":
      return [self.hyp_strengths]
    else:
      return []

  def set_params(self, params):
    if self.rec_model_type == "multinomial":
      trained_hyp_strengths = params[0]
      self.hyp_strengths.set_value(trained_hyp_strengths)
    else:
      return
    #avgs, cov_multiples = params
    #self.avgs.set_value(avgs)
    #self.cov_multiples.set_value(cov_multiples)
