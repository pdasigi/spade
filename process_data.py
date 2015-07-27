import codecs, re
import itertools
from nltk.corpus import wordnet as wn

class DataProcessor(object):
  def __init__(self):
    self.wanted_args = ['nsubj', 'dobj']

    self.thing_prons = ['which', 'that', 'this', 'what', 'these', 'itself', 'something', 'anything', 'everything'] # thing
    self.male_prons = ['he', 'him', 'himself', 'his'] # man.n.01
    self.female_prons = ['she', 'her', 'herself'] # woman.m.01
    self.people_prons = ['they', 'them', 'themselves', 'we', 'ourselves', 'yourselves'] # people.n.01, people.n.03
    self.person_prons = ['you', 'i', 'who', 'whom', 'whoever', 'anyone', 'everyone', 'myself', 'yourself'] # person.n.01
    
    self.thing_hypernyms = self.get_hypernyms_word('thing', syn_cutoff=4)
    self.man_hypernyms = self.get_hypernyms_syn(wn.synset('man.n.01'))
    self.woman_hypernyms = self.get_hypernyms_syn(wn.synset('woman.n.01'))
    self.people_hypernyms = self.get_hypernyms_syn(wn.synset('people.n.01'))
    self.loc_hypernyms = self.get_hypernyms_syn(wn.synset('geographical_area.n.01'))
    self.person_hypernyms = self.get_hypernyms_syn(wn.synset('person.n.01'))
    self.year_hypernyms = self.get_hypernyms_syn(wn.synset('year.n.01'))
    self.number_hypernyms = self.get_hypernyms_syn(wn.synset('number.n.01'))

    self.misc_hypernyms = set(self.loc_hypernyms).union(self.person_hypernyms)

  def get_hypernyms_syn(self, syn, path_cutoff=-1):
    hypernyms = []
    for path_list in syn.hypernym_paths():
      pruned_path_list = list(path_list) if path_cutoff == -1 or path_cutoff >= len(path_list) else [x for x in reversed(path_list)][:path_cutoff]
      hypernyms.extend([s.name() for s in pruned_path_list])
    return set(hypernyms)

  def get_hypernyms_word(self, word, pos='n', syn_cutoff=-1):
    hypernyms = []
    synsets = wn.synsets(word, pos=pos)
    pruned_synsets = list(synsets) if syn_cutoff == -1 else synsets[:syn_cutoff]
    for syn in pruned_synsets:
      hypernyms.extend(list(self.get_hypernyms_syn(syn, path_cutoff=5)))
    return set(hypernyms)

  def make_data(self, filename):
    datafile = codecs.open(filename, "r", "utf-8")
    #pred_hypernym_lens = []
    #arg_hypernym_lens = []
    x_data = []
    y_s_data = []
    word_hypernym_map = {}
    word_index = {}
    concept_index = {}
    for line in datafile:
      line_parts = line.strip().split('\t')
      pred = line_parts[0]
      pred_hypernyms = self.get_hypernyms_word(pred, pos='v', syn_cutoff=2)
      if len(pred_hypernyms) == 0:
        #print line.strip().encode('utf-8')
        continue
      labeled_args = {}
      slot_hypernyms = [(pred, pred_hypernyms)]
      #print pred
      for i in range(1, len(line_parts), 2):
        label, word = line_parts[i], line_parts[i+1]
        if label not in self.wanted_args:
          continue
        wrd_lower = word.lower()
        syns = wn.synsets(word)
        hypernyms = []
        if wrd_lower in self.thing_prons:
          hypernyms = list(self.thing_hypernyms)
        elif wrd_lower in self.male_prons:
          hypernyms = list(self.man_hypernyms)
        elif wrd_lower in self.female_prons:
          hypernyms = list(self.female_prons)
        elif wrd_lower in self.people_prons:
          hypernyms = list(self.people_hypernyms)
        elif wrd_lower in self.person_prons:
          hypernyms = list(self.person_hypernyms)
        elif len(syns) != 0:
          hypernyms = list(self.get_hypernyms_word(word, syn_cutoff=2))
        elif re.match('^[12][0-9]{3}$', word) is not None:
          # The argument looks like an year
          hypernyms = list(self.year_hypernyms)
        elif re.match('^[0-9,-]+', word) is not None:
          hypernyms = list(self.number_hypernyms)
        elif word[0].isupper():
          hypernyms = list(self.misc_hypernyms)
        if len(hypernyms) == 0:
          continue
        slot_hypernyms.append((word, hypernyms))
        labeled_args[label] = word
        #print label, word, hypernyms
        #arg_hypernym_lens.append(len(hypernyms))
      #pred_hypernym_lens.append(len(pred_hypernyms))
      if len(labeled_args) < len(self.wanted_args):
        # This means we did not get all the wanted args
        continue
      w_datum = [pred]
      #print slot_hypernyms, labeled_args, len(self.wanted_args), len(labeled_args) == len(self.wanted_args)
      
      if pred not in word_index:
        word_index[pred] = len(word_index)
      for w_label in self.wanted_args:
        l_arg = labeled_args[w_label]
        if l_arg not in word_index:
          word_index[l_arg] = len(word_index)
        w_datum.append(labeled_args[w_label])

      for w, h_list in slot_hypernyms:
        for h in h_list:
          if h not in concept_index:
            concept_index[h] = len(concept_index)
        if w not in word_hypernym_map:
          word_hypernym_map[w] = h_list
      
      x_data.append([word_index[x] for x in w_datum])
      w_hyp_inds = []
      for w in w_datum:
        w_hyps = word_hypernym_map[w]
        h_inds = [concept_index[y] for y in w_hyps]
        w_hyp_inds.append(h_inds)
      #y_s_datum = [[]]
      #for h_inds in w_hyp_inds:
      #  y_s_datum = [ i + [y] for y in h_inds for i in y_s_datum ]
      y_s_datum = [list(l) for l in itertools.product(*w_hyp_inds)]
      y_s_data.append(y_s_datum)        
    #print float(sum(pred_hypernym_lens))/len(pred_hypernym_lens)
    #print float(sum(arg_hypernym_lens))/len(arg_hypernym_lens)
    return x_data, y_s_data, word_index, concept_index, word_hypernym_map

#import sys
#dp = DataProcessor()
#x_data, y_s_data, w_ind, c_ind, w_h_map = dp.make_data(sys.argv[1])
#print "X data:"
#print x_data
#print "\nY_s data sizes:"
#print [len(y_d) for y_d in y_s_data]
#print "\nw_ind:"
#print w_ind
#print "\nc_ind:"
#print c_ind
#print "\nw_h_map:"
#print w_h_map
