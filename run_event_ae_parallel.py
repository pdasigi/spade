import sys
import time
import random
import cPickle
import codecs
import copy
import theano, numpy
import multiprocessing as mp

from theano import tensor as T
from event_ae import EventAE
from process_data import DataProcessor 

sys.setrecursionlimit(10000)
num_args = 2
num_slots = num_args + 1
hyp_hidden_size = 50
learning_rate = 0.01
wc_hidden_sizes = [50] * num_slots
cc_hidden_sizes = [50] * num_args
max_iter = 10

num_procs = int(sys.argv[2])

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
sanity_test_data = random.sample(train_data, len(train_data)/10)

vocab_size = len(w_ind)
ont_size = len(c_ind)
event_ae = EventAE(num_args, vocab_size, ont_size, hyp_hidden_size, wc_hidden_sizes, cc_hidden_sizes)
eaes = [copy.deepcopy(event_ae) for _ in range(num_procs)]
part_train_funcs = [eae.get_train_func(learning_rate) for eae in eaes]
#train_func = event_ae.get_train_func(learning_rate)
post_score_func = event_ae.get_posterior_func()

def synchronize_param():
  all_repr_params = [ [param.get_value() for param in eae.repr_params] for eae in eaes ]
  avg_repr_params = [numpy.mean(param, axis=0) for param in zip(*all_repr_params)]
  event_ae.set_repr_params(avg_repr_params)
  for eae in eaes:
    eae.set_repr_params(avg_repr_params)

  all_hyp_params = [ [param.get_value() for param in eae.hyp_model.get_params()] for eae in eaes ]
  avg_hyp_params = [numpy.mean(param, axis=0) for param in zip(*all_hyp_params)]
  event_ae.hyp_model.set_params(avg_hyp_params)
  for eae in eaes:
    eae.hyp_model.set_params(avg_hyp_params)
  
  all_worker_model_wcp_params = [ [[param.get_value() for param in wcp_model.get_params()] for wcp_model in eae.wc_pref_models] for eae in eaes]
  for model_num, all_model_wcp_params in enumerate(zip(*all_worker_model_wcp_params)):
    avg_wcp_model_params = []
    for param_num, all_wcp_params in enumerate(zip(*all_model_wcp_params)):
      # Averaging over all worker params
      avg_wcp_model_params.append(numpy.mean(all_wcp_params, axis=0))
    event_ae.wc_pref_models[model_num].set_params(avg_wcp_model_params)
    for eae in eaes:
      eae.wc_pref_models[model_num].set_params(avg_wcp_model_params)

  all_worker_model_ccp_params = [ [[param.get_value() for param in ccp_model.get_params()] for ccp_model in eae.cc_pref_models] for eae in eaes]
  for model_num, all_model_ccp_params in enumerate(zip(*all_worker_model_ccp_params)):
    avg_ccp_model_params = []
    for param_num, all_ccp_params in enumerate(zip(*all_model_ccp_params)):
      # Averaging over all worker params
      avg_ccp_model_params.append(numpy.mean(all_ccp_params, axis=0))
    event_ae.cc_pref_models[model_num].set_params(avg_ccp_model_params)
    for eae in eaes:
      eae.cc_pref_models[model_num].set_params(avg_ccp_model_params)
  
  all_rec_params = [ [param.get_value() for param in eae.rec_params] for eae in eaes ]
  avg_rec_params = [numpy.mean(param, axis=0) for param in zip(*all_rec_params)]
  event_ae.set_rec_params(avg_rec_params)
  for eae in eaes:
    eae.set_rec_params(avg_rec_params)

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
y_s = T.imatrix('y_s') # Matrix with all possible concept combinations

def train_on_data(part_train_func, train_data_part, proc_ind, costs=None):
  pool_costs = False
  if costs is None:
    costs = []
    pool_costs = True
  for dt_ind, (x_datum, y_s_datum) in enumerate(train_data_part):
    cost = part_train_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
    if pool_costs:
      costs.append(cost)
    else:
      costs.put(cost)
    if dt_ind % 10 == 0:
      print "Process %d trained on %d points"%(proc_ind, dt_ind)
  if pool_costs:
    return costs

for num_iter in range(max_iter):
  times = []
  random.shuffle(train_data)
  train_data_parts = []
  chunk_size = len(train_data) / num_procs
  for chunk_start_ind in range(0, len(train_data), chunk_size):
    train_data_parts.append(train_data[chunk_start_ind : chunk_start_ind + chunk_size])
  print >>sys.stderr, "Starting %d processes each with %d points to train"%(num_procs, chunk_size)
  costs_list = []
  starttime = time.time()
  #costs = mp.Queue()
  #processes = [mp.Process(target=train_on_data, args=(part_train_funcs[i], train_data_parts[i], i, costs)) for i in range(num_procs)]
  #for p in processes:
  #  p.start()
  #for p in processes:
  #  p.join()
  #while not costs.empty():
  #  costs_list.append(costs.get())
  pool = mp.Pool(processes=num_procs)
  results = [pool.apply_async(train_on_data, args=(part_train_funcs[i], train_data_parts[i], i)) for i in range(num_procs)]
  for r in results:
    costs_list.extend(r.get())
  endtime = time.time()
  zero_costs = sum([1 if c == 0.0 else 0 for c in costs_list])
  avg_cost = sum(costs_list)/len(costs_list)
  print >>sys.stderr, "Finished iteration %d.\n\tAverage train cost: %r\n\tTime %d sec\n\tZero costs: %d"%(num_iter + 1, avg_cost, endtime-starttime, zero_costs)
  
  print >>sys.stderr, "Synchronizing param"
  synchronize_param()
  print >>sys.stderr, "Done synchronizing. Dumping param."
  
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

