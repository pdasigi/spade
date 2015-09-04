import theano, numpy
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#from theano.tensor.shared_randomstreams import RandomStreams
from theano.ifelse import ifelse

from hypernymy import HypernymModel
from preferences import PreferenceModel
from reconstruction import ReconstructionModel 

SMALL_NUM = 1e-30
LOG_SMALL_NUM = numpy.log(SMALL_NUM)

class EventAE(object):
  def __init__(self, num_args, vocab_size, ont_size, hyp_hidden_size, wc_hidden_sizes, cc_hidden_sizes, word_dim=50, concept_dim=50):
    numpy_rng = numpy.random.RandomState(12345)
    self.theano_rng = RandomStreams(12345)
    self.ont_size = ont_size
    vocab_rep_range = 4 * numpy.sqrt(6. / (vocab_size + word_dim))
    init_vocab_rep = numpy.asarray(numpy_rng.uniform(low = -vocab_rep_range, high = vocab_rep_range, size=(vocab_size, word_dim)) )
    ont_rep_range = 4 * numpy.sqrt(6. / (ont_size + concept_dim))
    init_ont_rep = numpy.asarray(numpy_rng.uniform(low = -ont_rep_range, high = ont_rep_range, size=(ont_size, concept_dim)) )
    vocab_rep = theano.shared(value=init_vocab_rep, name='vocab_rep')
    ont_rep = theano.shared(value=init_ont_rep, name='ont_rep')
    self.repr_params = [vocab_rep, ont_rep]
    self.enc_params = []
    self.hyp_model = HypernymModel('linlayer', hyp_hidden_size, vocab_rep, ont_rep)
    self.enc_params.extend(self.hyp_model.get_params())
    self.wc_pref_models = []
    self.cc_pref_models = []
    self.num_slots = num_args + 1 # +1 for the predicate
    self.num_args = num_args
    for i in range(self.num_slots):
      wc_pref_model = PreferenceModel('word_concept', 'linlayer', wc_hidden_sizes[i], ont_rep, vocab_rep)
      self.wc_pref_models.append(wc_pref_model)
      self.enc_params.extend(wc_pref_model.get_params())
  
    for i in range(num_args):
      cc_pref_model = PreferenceModel('concept_concept', 'linlayer', cc_hidden_sizes[i], ont_rep)
      self.cc_pref_models.append(cc_pref_model)
      self.enc_params.extend(cc_pref_model.get_params())
    self.rec_model = ReconstructionModel(ont_size, vocab_rep)
    self.rec_params = self.rec_model.get_params()
    # Random y, sampled from uniform(|ont|^num_slots)
    self.y_r = T.cast(self.theano_rng.uniform(low=0, high=self.ont_size-1, size=(self.num_slots,)), 'int32')
    self.num_enc_ns = 1
    self.num_label_ns = 1
  
  def get_sym_rand_y(self, y_s):
    # NCE function
    # Sample randomly from y|x
    rand_ind = T.cast(self.theano_rng.uniform(low=0, high=y_s.shape[0]-1, size=(1,)), 'int32')
    sample = y_s[rand_ind[0]]
    return sample
    
  def get_sym_encoder_energy(self, x, y):
    # Works with NCE
    hsum = T.constant(0)
    for i in range(self.num_slots):
      hsum += self.hyp_model.get_symb_score(x[i], y[i])
    p_w_c_sum = T.constant(0)
    for i in range(self.num_slots):
      for j in range(self.num_slots):
        if i == j:
          continue
        p_w_c_sum += self.wc_pref_models[i].get_symb_score(x[i], y[j])
    p_c_c_sum = T.constant(0)
    for i in range(self.num_args):
      p_c_c_sum += self.cc_pref_models[i].get_symb_score(y[0], y[i + 1])
    return hsum + p_w_c_sum + p_c_c_sum

  def get_sym_encoder_partition(self, x, y_s):
    partial_sums, _ = theano.scan(fn=lambda y, interm_sum, x_0: interm_sum + T.exp(self.get_sym_encoder_energy(x_0, y)), outputs_info=numpy.asarray(0.0, dtype='float64'), sequences=[y_s], non_sequences=x)
    encoder_partition = partial_sums[-1]
    return encoder_partition

  def get_sym_nc_encoder_prob(self, x, y, y_s, num_noise_samples=None):
    # NCE function
    # Noise distribution is not conditioned on x. So we sample directly from ont, not from y_s
    if num_noise_samples is None:
      num_noise_samples = self.num_enc_ns
    enc_energy = T.exp(self.get_sym_encoder_energy(x, y))
    ns_prob = num_noise_samples * ((1. / self.ont_size) ** self.num_slots)
    true_prob = enc_energy / (enc_energy + ns_prob)
    noise_prob = T.constant(1.0, dtype='float64')
    for _ in range(num_noise_samples):
      ns_enc_energy = self.get_sym_encoder_energy(x, self.y_r)
      #ns_enc_energy = T.exp(self.get_sym_encoder_energy(x, self.get_sym_rand_y(y_s)))
      noise_prob *= ns_prob / (ns_enc_energy + ns_prob)
    return true_prob * noise_prob

  def get_sym_rec_prob(self, x, y):
    # Works with NCE
    init_prob = T.constant(1.0, dtype='float64')
    partial_prods, _ = theano.scan(fn = lambda x_i, y_i, interm_prod: interm_prod * self.rec_model.get_sym_rec_prob(x_i, y_i), outputs_info=init_prob, sequences=[x, y])
    rec_prob = partial_prods[-1]
    return rec_prob
    
  def get_sym_posterior_num(self, x, y):
    # Needed for NCE
    enc_energy = self.get_sym_encoder_energy(x, y)
    rec_prob = self.get_sym_rec_prob(x, y)
    return T.exp(enc_energy) * rec_prob
      
  def get_sym_posterior_partition(self, x, y_s):
    partial_sums, _ = theano.scan(fn=lambda y, interm_sum, x_0: interm_sum + self.get_sym_posterior_num(x_0, y), outputs_info=numpy.asarray(0.0, dtype='float64'), sequences=[y_s], non_sequences=x)
    posterior_partition = partial_sums[-1]
    return posterior_partition

  def get_sym_nc_posterior(self, x, y, y_s, num_noise_samples=None):
    # NCE function
    # p(\hat{x}, y | x)
    if num_noise_samples is None:
      num_noise_samples = self.num_enc_ns
    return self.get_sym_nc_encoder_prob(x, y, y_s, num_noise_samples=num_noise_samples) * self.get_sym_rec_prob(x, y)

  def get_sym_nc_label_prob(self, x, y, y_s, num_noise_samples=None):
    # NCE function
    # p(y | x, \hat{x})
    if num_noise_samples is None:
      num_noise_samples = self.num_label_ns
    true_posterior = self.get_sym_nc_posterior(x, y)
    #TODO: Can make this more efficient
    noise_posterior = self.get_sym_nc_posterior(x, self.get_sym_rand_y(y_s), num_noise_samples=1)
    ns_prob = num_noise_samples * T.pow(1. / y_s.shape[0], self.num_slots)
    true_prob = true_posterior / (true_posterior + ns_prob)
    noise_prob = T.constant(1.0, dtype='float64')
    for _ in range(num_noise_samples):
      #noise_posterior = self.get_sym_nc_posterior(x, self.get_sym_rand_y(y_s))
      noise_prob *= ns_prob / (noise_posterior + ns_prob)
    return true_prob * noise_prob

  def get_sym_complete_expectation(self, x, y_s):
    encoder_partition = self.get_sym_encoder_partition(x, y_s)
    posterior_partition = self.get_sym_posterior_partition(x, y_s)
    def prod_fun(y_0, interm_sum, x_0): 
      post_num = self.get_sym_posterior_num(x_0, y_0)
      fixed_post_num = ifelse(T.le(post_num, SMALL_NUM), T.constant(0.0, dtype='float64'), post_num)
      return interm_sum + ifelse(T.le(fixed_post_num, SMALL_NUM), T.constant(0.0, dtype='float64'), fixed_post_num * T.log(fixed_post_num))
    partial_sums, _ = theano.scan(fn=prod_fun, outputs_info=numpy.asarray(0.0, dtype='float64'), sequences=[y_s], non_sequences=x)
    data_term = ifelse(T.eq(posterior_partition, T.constant(0.0, dtype='float64')), T.constant(0.0, dtype='float64'), partial_sums[-1] / posterior_partition)
    #data_term = partial_sums[-1]
    complete_expectation = data_term - T.log(encoder_partition)
    #complete_expectation = data_term
    return complete_expectation
  
  def get_sym_nc_complete_expectation(self, x, y_s):
    # NCE function
    def get_expectation(y_0, interm_sum, x_0, Y):
      label_prob = self.get_sym_nc_label_prob(x_0, y_0, Y)
      posterior = self.get_sym_nc_posterior(x_0, y_0)
      log_posterior = ifelse(T.le(posterior, SMALL_NUM), T.constant(LOG_SMALL_NUM, dtype='float64'), T.log(posterior))
      return interm_sum + (label_prob * log_posterior)
    res, _ = theano.scan(fn=get_expectation, outputs_info=numpy.asarray(0.0, dtype='float64'), sequences=[y_s], non_sequences=[x, y_s])
    complete_expectation = res[-1]
    return complete_expectation
  
  def get_sym_nc_direct_prob(self, x, y_s):
    # NCE function
    def get_prob(y_0, interm_sum, x_0, Y):
      posterior = self.get_sym_nc_posterior(x_0, y_0, Y)
      return interm_sum + posterior
    res, _ = theano.scan(fn=get_prob, outputs_info=numpy.asarray(0.0, dtype='float64'), sequences=[y_s], non_sequences=[x, y_s])
    direct_prob = res[-1]
    return direct_prob
    
  def get_train_func(self, learning_rate, nce=True):
    # TODO: Implement AdaGrad
    x, y_s = T.ivector("x"), T.imatrix("y_s")
    #em_cost = -self.get_sym_nc_complete_expectation(x, y_s) if nce else -self.get_sym_complete_expectation(x, y_s)
    cost = -self.get_sym_nc_direct_prob(x, y_s)
    params = self.repr_params + self.enc_params + self.rec_params
    g_params = T.grad(cost, params)
    # Updating the parameters only if the norm of the gradient is less than 100.
    # Important: This check also takes care of any element in the gradients being nan. The conditional returns False even in that case.
    updates=[ (p, ifelse(T.le(T.nlinalg.norm(g, None), T.constant(100.0, dtype='float64')), p - learning_rate * g, p)) for p, g in zip(params, g_params) ]
    train_func = theano.function([x, y_s], cost, updates=updates)
    return train_func

  def get_posterior_func(self):
    # Works with NCE
    x, y = T.ivectors('x', 'y')
    posterior_func = theano.function([x, y], self.get_sym_posterior_num(x, y))
    return posterior_func

  def set_repr_params(self, repr_param_vals):
    for i, param_val in enumerate(repr_param_vals):
      self.repr_params[i].set_value(param_val)

  def set_rec_params(self, rec_param_vals):
    for i, param_val in enumerate(rec_param_vals):
      self.rec_params[i].set_value(param_val)

