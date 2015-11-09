import sys
import argparse
import time
import random
import cPickle
import gzip
import copy
import codecs
import operator
import multiprocessing as mp

import theano, numpy
from theano import tensor as T
from event_ae import EventAE
from process_data import DataProcessor 

argparser = argparse.ArgumentParser(description="Run Selectional Preference AutoencoDEr")
argparser.add_argument('train_file', metavar="TRAIN-FILE", type=str, help="File containing n-tuples to train on")
argparser.add_argument('word_types', metavar="WORD-TYPES", type=str, help="String showing the WordNet POS types of the words in the train file, separated by _. Eg. a_n for adj-noun, v_n_n for verb-noun-noun etc.")
argparser.add_argument('--pt_rep', type=str, help="File containing pretrained embeddings")
argparser.add_argument('--change_word_rep', help="Make changes to word representations (Default is False)", action='store_true')
argparser.set_defaults(change_word_rep=False)
argparser.add_argument('--word_dim', type=int, help="Dimensionality of word representations", default=50)
argparser.add_argument('--concept_dim', type=int, help="Dimensionality of concept representations", default=50)
argparser.add_argument('--write_model_freq', type=int, help="Frequency at which the model will be written to disk", default=1)
argparser.add_argument('--num_slots', type=int, help="Number of slots in the input", default=2)
argparser.add_argument('--hyp_hidden_size', type=int, help="Hidden layer size in hypernymy if a NN factor is selected", default=20)
argparser.add_argument('--wc_hidden_size', type=int, help="Hidden layer size in word-concept preferences if a NN factor is selected", default=20)
argparser.add_argument('--wc_lr_wp_rank', type=int, help="Rank of the low rank weights for word-concept preferences if low rank weighted inner product is chosen", default=10)
argparser.add_argument('--cc_hidden_size', type=int, help="Hidden layer size in concept-concept preferences if a NN factor is selected", default=20)
argparser.add_argument('--cc_lr_wp_rank', type=int, help="Rank of the low rank weights for concept-concept preferences if low rank weighted inner product is chosen", default=10)
argparser.add_argument('--lr', type=float, help="Learning rate", default=0.01)
argparser.add_argument('--max_iter', type=int, help="Maximum number of iterations", default=100)
argparser.add_argument('--vocab_file', type=str, help="Word vocabulary file", default="vocab.txt")
argparser.add_argument('--ont_file', type=str, help="Concept vocabulary file", default="ont.txt")
argparser.add_argument('--parallel', type=int, help="Number of processors to run on (default 1)", default=1)
argparser.add_argument('--use_relaxation', help="Ignore inter-concept preferences and optimize", action='store_true')
argparser.set_defaults(use_relaxation=False)
argparser.add_argument('--use_em', help="Use EM (Default is False)", action='store_true')
argparser.set_defaults(use_em=False)
argparser.add_argument('--use_nce', help="Use NCE for estimating encoding probability. (Default is False)", action='store_true')
argparser.set_defaults(use_nce=False)
argparser.add_argument('--hyp_model_type', type=str, help="Hypernymy model (weighted_prod, linlayer, tanhlayer)", default="weighted_prod")
argparser.add_argument('--wc_pref_model_type', type=str, help="Word-concept preference model (weighted_prod, linlayer, tanhlayer)", default="tanhlayer")
argparser.add_argument('--cc_pref_model_type', type=str, help="Concept-concept preference model (weighted_prod, linlayer, tanhlayer)", default="tanhlayer")
args = argparser.parse_args()
pred_arg_pos = args.word_types.split("_")
learning_rate = args.lr
use_pretrained_wordrep = False 
if args.pt_rep:
  use_pretrained_wordrep = True
  pt_word_rep = {l.split()[0]: numpy.asarray([float(f) for f in l.strip().split()[1:]]) for l in gzip.open(args.pt_rep)}

dp = DataProcessor(pred_arg_pos)
x_data, y_s_data, w_ind, c_ind, w_h_map, w_oov, c_oov = dp.make_data(args.train_file, relaxed=args.use_relaxation)
rev_w_ind = {ind:word for word, ind in w_ind.items()}
rev_c_ind = {ind:concept for concept, ind in c_ind.items()}

if len(w_oov) != 0:
  print >>sys.stderr, "Regarding %d words as OOV"%(len(w_oov))

if len(c_oov) != 0:
  print >>sys.stderr, "Regarding %d concepts as OOV"%(len(c_oov))

if not args.use_relaxation:
  num_slots = len(x_data[0])
else:
  num_slots = args.num_slots

num_args = num_slots - 1

wc_hidden_sizes = [args.wc_hidden_size] * num_slots
cc_hidden_sizes = [args.cc_hidden_size] * num_args

vocab_file = codecs.open(args.vocab_file, "w", "utf-8")
for w, ind in w_ind.items():
  print >>vocab_file, w, ind
vocab_file.close()

ont_file = codecs.open(args.ont_file, "w", "utf-8")
for c, ind in c_ind.items():
  print >>ont_file, c, ind
ont_file.close()

train_data = zip(x_data, y_s_data)
sanity_test_data = random.sample(train_data, min(len(train_data)/10, 20))

num_cands = sum([len(y_s_d) for _, y_s_d in train_data])
print >>sys.stderr, "Read training data. Average number of concept candidates per point: %f"%(float(num_cands)/len(train_data))

vocab_size = len(w_ind)
ont_size = len(c_ind)

comp_starttime = time.time()
event_ae = EventAE(num_args, vocab_size, ont_size, args.hyp_hidden_size, wc_hidden_sizes, cc_hidden_sizes, word_dim=args.word_dim, concept_dim=args.concept_dim, word_rep_param=args.change_word_rep, hyp_model_type=args.hyp_model_type, wc_pref_model_type=args.wc_pref_model_type, cc_pref_model_type=args.cc_pref_model_type, relaxed=args.use_relaxation, wc_lr_wp_rank=args.wc_lr_wp_rank, cc_lr_wp_rank=args.cc_lr_wp_rank)

if use_pretrained_wordrep:
  num_words_covered = 0
  init_wordrep = event_ae.vocab_rep.get_value()
  for word in w_ind:
    if word in pt_word_rep:
      init_wordrep[w_ind[word]] = pt_word_rep[word]
      num_words_covered += 1
  print >>sys.stderr, "Using pretrained word representations from %s"%(args.pt_rep)
  print >>sys.stderr, "\tcoverage is %f"%(float(num_words_covered)/len(w_ind))

if args.parallel != 1:
  eaes = [copy.deepcopy(event_ae) for _ in range(args.parallel)]
  if args.use_relaxation: 
    part_train_funcs = [[eae.get_relaxed_train_func(learning_rate, s) for s in range(num_slots)] for eae in eaes]
  else:
    part_train_funcs = [eae.get_train_func(learning_rate) for eae in eaes]
else:
  if args.use_relaxation:
    train_funcs = [event_ae.get_relaxed_train_func(learning_rate, s) for s in range(num_slots)]
    post_score_funcs = [event_ae.get_relaxed_posterior_func(s) for s in range(num_slots)]
  else:
    train_func = event_ae.get_train_func(learning_rate, em=args.use_em, nce=args.use_nce)
    post_score_func = event_ae.get_posterior_func()

comp_endtime = time.time()
print >>sys.stderr, "Theano compilation took %d seconds"%(comp_endtime - comp_starttime)

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

  if not args.use_relaxation:
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
    if args.use_relaxation:
      s = x_datum[-1]
      pscore_func = post_score_funcs[s]
      score = pscore_func(numpy.asarray(x_datum[:-1], dtype='int32'), numpy.asarray(y_datum, dtype='int32'))
    else:
      global post_score_func
      score = post_score_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_datum, dtype='int32'))
    if score > max_score:
      max_score = score
      best_y = y_datum
  return best_y, max_score

def train_on_data(part_train_func, train_data_part, proc_ind, costs=None):
  pool_costs = False
  if costs is None:
    costs = []
    pool_costs = True
  for dt_ind, (x_datum, y_s_datum) in enumerate(train_data_part):
    if args.use_relaxation:
      cost = part_train_func[x_datum[-1]](numpy.asarray(x_datum[:-1], dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
    else:
      cost = part_train_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
    if pool_costs:
      costs.append(cost)
    else:
      costs.put(cost)
    if dt_ind % 1000 == 0:
      print "Process %d trained on %d points"%(proc_ind, dt_ind)
  if pool_costs:
    return costs

print >>sys.stderr, "Starting training"
for num_iter in range(args.max_iter):
  costs = []
  times = []
  random.shuffle(train_data)
  epoch_starttime = time.time()
  if args.parallel != 1:
    train_data_parts = []
    chunk_size = len(train_data) / args.parallel
    for chunk_start_ind in range(0, len(train_data), chunk_size):
      train_data_parts.append(train_data[chunk_start_ind : chunk_start_ind + chunk_size])
    print >>sys.stderr, "Starting %d processes each with %d points to train"%(args.parallel, chunk_size)
    costs = []
    pool = mp.Pool(processes=args.parallel)
    results = [pool.apply_async(train_on_data, args=(part_train_funcs[i], train_data_parts[i], i)) for i in range(args.parallel)]
    for r in results:
      costs.extend(r.get())
  else:
    for i, (x_datum, y_s_datum) in enumerate(train_data):
      inst_starttime = time.time()
      if args.use_relaxation:
        train_func = train_funcs[x_datum[-1]]
        cost = train_func(numpy.asarray(x_datum[:-1], dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
      else:
        cost = train_func(numpy.asarray(x_datum, dtype='int32'), numpy.asarray(y_s_datum, dtype='int32'))
      times.append(time.time() - inst_starttime)
      costs.append(cost)
      if (i+1) % 1000 == 0:
        print >>sys.stderr, "Processed %d points.  Average cost till now is %r"%(i, sum(costs)/len(costs))
        print >>sys.stderr, "\tAverage time per point till now is %f sec"%(float(sum(times))/len(times))

  avg_cost = sum(costs)/len(costs)
  epoch_endtime = time.time()
  print >>sys.stderr, "Finished iteration %d.\n\tAverage train cost: %f\n\tTime %d sec"%(num_iter + 1, avg_cost, epoch_endtime-epoch_starttime)
  if args.parallel != 1:
    print >>sys.stderr, "Synchronizing param"
    synchronize_param()
    print >>sys.stderr, "Done synchronizing. Dumping param."

  if (num_iter+1) % args.write_model_freq == 0:
    repr_param_out = open("repr_params_%d.pkl"%(num_iter + 1), "wb")
    #repr_params = [param.get_value() for param in event_ae.repr_params]
    repr_params = [event_ae.vocab_rep.get_value(), event_ae.ont_rep.get_value()]
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
    if not args.use_relaxation:
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
      x_words = [rev_w_ind[x_ind] for x_ind in x_datum[:-1]] if args.use_relaxation else [rev_w_ind[x_ind] for x_ind in x_datum]
      best_y_ind, best_score = get_mle_y(x_datum, y_s_datum)
      if args.use_relaxation:
        y_concepts = [rev_c_ind[best_y_ind]]
      else:
        y_concepts = [rev_c_ind[ind] for ind in best_y_ind]
      print >>sys.stderr, "x: %s"%(" ".join(x_words))
      print >>sys.stderr, "y: %s"%(" ".join(y_concepts))
