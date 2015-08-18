import sys
import time
import random
import cPickle
import codecs
import theano, numpy
from theano import tensor as T
from event_ae import EventAE
from process_data import DataProcessor 

num_args = 2
num_slots = num_args + 1
hyp_hidden_size = 50
learning_rate = 0.01
wc_hidden_sizes = [50] * num_slots
cc_hidden_sizes = [50] * num_args
max_iter = 10

dp = DataProcessor()
x_data, y_s_data, w_ind, c_ind, w_h_map = dp.make_data(sys.argv[1])

vocab_file = codecs.open("vocab.txt", "w", "utf-8")
for w, ind in w_ind.items():
  print >>vocab_file, w, ind
vocab_file.close()

ont_file = codecs.open("ont.txt", "w", "utf-8")
for c, ind in c_ind.items():
  print >>ont_file, c, ind
ont_file.close()

rev_w_ind = {ind:word for word, ind in w_ind.items()}
rev_c_ind = {ind:concept for concept, ind in c_ind.items()}
train_data = zip(x_data, y_s_data)
sanity_test_data = train_data[20:70]

vocab_size = len(w_ind)
ont_size = len(c_ind)
event_ae = EventAE(num_args, vocab_size, ont_size, hyp_hidden_size, wc_hidden_sizes, cc_hidden_sizes)
train_func = event_ae.get_train_func(learning_rate)
post_score_func = event_ae.get_posterior_func()

def get_mle_y(x_datum, y_s_datum):
  max_score = -float("inf")
  best_y = []
  for y_datum in y_s_datum:
    score = post_score_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_datum, dtype='int32'))
    if score > max_score:
      max_score = score
      best_y = y_datum
  return best_y, max_score

x = T.ivector('x') # Vector of word indexes in vocabulary
#y = T.ivector('y') # Vector of concept indexes in ontology
y_s = T.imatrix('y_s') # Matrix with all possible concept combinations
#post_num = event_ae.get_sym_posterior_num(x, y)
#enc_energy = event_ae.get_sym_encoder_energy(x, y)
#enc_partition = event_ae.get_sym_encoder_partition(x, y_s)
#post_part = event_ae.get_sym_posterior_partition(x, y_s)
#rec_prob = event_ae.get_sym_rec_prob(x, y)
comp_ex = event_ae.get_sym_complete_expectation(x, y_s)
#f = theano.function([x, y], T.exp(enc_energy))
#f1 = theano.function([x, y_s], enc_partition)
#f2 = theano.function([x, y], post_num)
#f4 = theano.function([x, y], rec_prob)
f3 = theano.function([x, y_s], comp_ex)
#f5 = theano.function([x, y_s], post_part)
#x_g1 = numpy.asarray([0, 2, 2], dtype='int32')
#x_g2 = numpy.asarray([0, 2, 3], dtype='int32')
#x_g3 = numpy.asarray([0, 3, 4], dtype='int32')
#x_g4 = numpy.asarray([0, 4, 5], dtype='int32')
#y_g = numpy.asarray([1, 0, 0], dtype='int32')
#print f(x_g1, y_g)
#print f2(x_g1, y_g)
#print f4(x_g1, y_g)


for num_iter in range(max_iter):
  costs = []
  times = []
  num_cands = []
  random.shuffle(train_data)
  #print >>sys.stderr, "x_data", [x for x, y in train_data]
  ign_points = 0
  starttime = time.time()
  for i, (x_datum, y_s_datum) in enumerate(train_data):
    num_cands.append(len(y_s_datum))
    c_ex = f3(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
    if numpy.isnan(c_ex):
      """print >>sys.stderr, "Potential nan for datum:", " ".join([rev_w_ind[ind] for ind in x_datum])
      ep = f1(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
      print >>sys.stderr, "enc_partition:", ep
      pp = f5(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32')) 
      print >>sys.stderr, "post_partition:", pp
      if not numpy.isnan(ep) and not numpy.isnan(pp):
        for y_datum in y_s_datum:
          print >>sys.stderr, "rec_prob:", f4(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_datum, dtype='int32'))
      else: 
        if numpy.isnan(ep):
          conc_enc_part = 0.0
          for y_datum in y_s_datum:
            exp_enc_en = f(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_datum, dtype='int32'))
            print >>sys.stderr, "exp (enc energy):", exp_enc_en
            conc_enc_part += exp_enc_en
          print >>sys.stderr, "concrete partition:", conc_enc_part
        if numpy.isnan(pp):
          for y_datum in y_s_datum:
            print >>sys.stderr, "post num:", f2(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_datum, dtype='int32'))
      print >>sys.stderr, "Paused. Press something..."
      sys.stdin.readline()"""
      ign_points += 1
      continue
    cost = train_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
    costs.append(cost)
  endtime = time.time()
  avg_cost = sum(costs)/len(costs)
  print >>sys.stderr, "Finished iteration %d.\n\tAverage train cost: %f\n\tTime %d sec\n\tIgnored %d points"%(num_iter + 1, avg_cost, endtime-starttime, ign_points)
  repr_param_out = open("repr_params_%d.pkl"%(num_iter + 1), "wb")
  repr_params = [param.get_value() for param in event_ae.repr_params]
  cPickle.dump(repr_params, repr_param_out)
  repr_param_out.close()
  hyp_param_out = open("hyp_params_%d.pkl"%(num_iter + 1), "wb")
  hyp_params = [param.get_value() for param in event_ae.hyp_model.get_params()]
  cPickle.dump(hyp_params, hyp_param_out)
  hyp_param_out.close()
  wcp_param_out = open("wcp_params_%d.pkl"%(num_iter + 1), "wb")
  wcp_params = [[param.get_value() for param in wcp_model.get_params()] for wcp_model in event_ae.wc_pref_models]
  cPickle.dump(wcp_params, wcp_param_out)
  wcp_param_out.close()
  ccp_param_out = open("ccp_params_%d.pkl"%(num_iter + 1), "wb")
  ccp_params = [[param.get_value() for param in ccp_model.get_params()] for ccp_model in event_ae.cc_pref_models]
  cPickle.dump(ccp_params, ccp_param_out)
  ccp_param_out.close()
  rec_param_out = open("rec_params_%d.pkl"%(num_iter + 1), "wb")
  rec_params = [param.get_value() for param in event_ae.rec_params]
  cPickle.dump(rec_params, rec_param_out)
  rec_param_out.close()
  #print >>sys.stderr, "Costs:", costs
  #print f(x_g1, y_g)
  #print f2(x_g1, y_g)
  #print f4(x_g1, y_g)
  print >>sys.stderr, "Sanity test output:"
  for (x_datum, y_s_datum) in sanity_test_data:
    x_words = [rev_w_ind[x_ind] for x_ind in x_datum]
    best_y_ind, best_score = get_mle_y(x_datum, y_s_datum)
    y_concepts = [rev_c_ind[ind] for ind in best_y_ind]
    print >>sys.stderr, "x: %s"%(" ".join(x_words))
    print >>sys.stderr, "y: %s"%(" ".join(y_concepts))


#y_s_g = numpy.asarray([[1, 0, 0], [0, 0, 0], [0, 1, 1]], dtype='int32')
#print f(x_g1, y_g)
#print f1(x_g1, y_s_g)
#for x_datum, y_s_datum in sanity_test_data:
#  for y_datum in y_s_datum:
#    prob = f4(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_datum, dtype='int32'))
#    num = f2(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_datum, dtype='int32'))
#    print prob, num
#print f3(x_g1, y_s_g)
#print f5(x_g1, y_s_g)
#print train_func(x_g1, y_s_g)
#print train_func(x_g2, y_s_g)
#print train_func(x_g3, y_s_g)
#print train_func(x_g4, y_s_g)
#x_i, y_i = T.iscalars(2)
#f6 = theano.function([x_i, y_i], event_ae.rec_model.get_sym_rec_prob(x_i, y_i))
#for i in range(3):
#  print f6(x_g1[i], y_g[i])
