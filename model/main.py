import json
import random
from learner import setupLearner
import examples
import time

with open('params.json', 'r') as f:
    cfg = json.load(f)


# Load data
orthography = examples.load('../inputs/orth.csv', simplify=True)
phonology = examples.load('../inputs/phon.csv', simplify=True)
words = [x.strip() for x in open('../inputs/words.csv','r').readlines()]

T = []

start_time = time.time()
training_environment = random.sample(range(len(orthography)), cfg['training_size'])
T.append(training_environment)
learner, model = setupLearner(
    hidden_layer_sizes=cfg['hidden_size'],
    max_iter=cfg['max_iter'],
    input_patterns=orthography,
    target_patterns=phonology,
    test_set = training_environment)
learner.fit(training_environment)

end_time = time.time()
print(round((end_time-start_time)/60, 2), "minutes elapsed since start of learning")



dateID = time.asctime().replace(' ', '_')

with open(str('../outputs/' + dateID + '.csv'), 'w') as f:
    for i,w in enumerate(words):
        f.write("{word:s},".format(word=w))
        isTrainingItem = [i in T[j] for j in range(len(T))]
        f.write("{vec:s}\n".format(vec = ','.join([str(int(tf)) for tf in isTrainingItem])))
