import sys
import time
import random
import cPickle
import codecs
import theano, numpy
from theano import tensor as T
from event_ae import EventAE
from process_data import DataProcessor 

pred_arg_pos = sys.argv[2].split("_")
dp = DataProcessor(pred_arg_pos)
x_data, y_s_data, w_ind, c_ind, w_h_map = dp.make_data(sys.argv[1])

num_slots = len(x_data[0])
num_args = num_slots - 1
hyp_hidden_size = 20
learning_rate = 0.01
wc_hidden_sizes = [20] * num_slots
cc_hidden_sizes = [20] * num_args
max_iter = 10

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
sanity_test_data = random.sample(train_data, min(len(train_data)/10, 20))

num_cands = sum([len(y_s_d) for _, y_s_d in train_data])
print >>sys.stderr, "Read training data. Average number of concept candidates per point: %f"%(float(num_cands)/len(train_data))

vocab_size = len(w_ind)
ont_size = len(c_ind)

comp_starttime = time.time()
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
comp_endtime = time.time()
print >>sys.stderr, "Theano compilation took %d seconds"%(comp_endtime - comp_starttime)

x = T.ivector('x') # Vector of word indexes in vocabulary
y_s = T.imatrix('y_s') # Matrix with all possible concept combinations

print >>sys.stderr, "Starting training"
for num_iter in range(max_iter):
  costs = []
  times = []
  random.shuffle(train_data)
  epoch_starttime = time.time()
  for i, (x_datum, y_s_datum) in enumerate(train_data):
    inst_starttime = time.time()
    cost = train_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
    times.append(time.time() - inst_starttime)
    costs.append(cost)
    if i % 100 == 0:
      print >>sys.stderr, "Processed %d points.  Average cost till now is %r"%(i, sum(costs)/len(costs))
      print >>sys.stderr, "\tAverage time per point till now is %f sec"%(float(sum(times))/len(times))
  epoch_endtime = time.time()
  avg_cost = sum(costs)/len(costs)
  print >>sys.stderr, "Finished iteration %d.\n\tAverage train cost: %f\n\tTime %d sec"%(num_iter + 1, avg_cost, epoch_endtime-epoch_starttime)
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
  print >>sys.stderr, "Sanity test output:"
  for (x_datum, y_s_datum) in sanity_test_data:
    x_words = [rev_w_ind[x_ind] for x_ind in x_datum]
    best_y_ind, best_score = get_mle_y(x_datum, y_s_datum)
    y_concepts = [rev_c_ind[ind] for ind in best_y_ind]
    print >>sys.stderr, "x: %s"%(" ".join(x_words))
    print >>sys.stderr, "y: %s"%(" ".join(y_concepts))
