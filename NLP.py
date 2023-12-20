import sys, random, math
from collections import Counter
import numpy as np

np.random.seed(1)   # Save random generation
random.seed(1)


f = open('path.../ reviews.txt')    # Open file
raw_reviews = f.readlines()
f.close()

tokens = list(map(lambda x: (x.split(" ")), raw_reviews))       # set | split for every iter items  [[/n], [/n]]
word_cnt = Counter()

for sent in tokens:

    for word in sent:
        word_cnt[word] -= 1

vocab = list(set(map(lambda x:x[0],word_cnt.most_common())))         # List with most common words in reviews  [ "word", "word" ... ]

word2index = {}

for i,word in enumerate(vocab):         # word2index {element : index}
    word2index[word] = i

concatenated = list()
input_dataset = list()

for sent in tokens:
    sent_indices = list()

    for word in sent:

        try:

            sent_indices.append(word2index[word])        # That's gonna be a 2 variables
            concatenated.append(word2index[word])        # with indices

        except:

            ""

    input_dataset.append(sent_indices)         # And this is our input data. Yeah, every index - input data!

concatenated = np.array(concatenated)

# >>> Data for training

random.shuffle(input_dataset)

alpha, iterations = (0.05, 2)
hidden_size, window, negative = (50, 2, 5)

# >>> Weights

weights_0_1 = (np.random.rand(len(vocab), hidden_size) - 0.5) * 0.2
weights_1_2 = np.random.rand(len(vocab), hidden_size) * 0

# >>> One hot labels for target word

layer_2_target = np.zeros(negative + 1)
layer_2_target[0] = 1

# >>> Finding the same words to target word

def similiar(target= 'beatiful'):

    target_index = word2index[target]

    scores = Counter()

    for word, index in word2index.items():

        raw_difference = weights_0_1[index] - (weights_1_2[target_index])
        squad_difference = raw_difference * raw_difference
        scores[word] = -math.sqrt(sum(squad_difference))

    return scores.most_common(10)

# >>> sigmoid function

def sigmoid(x):

    return 1/ (1 + np.exp(-x))

# >>> Word2Vec   -  CBOW

for rev_i, review in enumerate(input_dataset * iterations):

    for target_i in range(len(review)):

        target_samples = [review[target_i]] + list(concatenated[(np.random.rand(negative) * len(concatenated)).astype('int').tolist()])

        left_context = review[max(0, target_i - window) : target_i]
        right_context = review[target_i + 1 : min(len(review), target_i + window)]

# >>> First and second layers

layer_1 = np.mean(weights_0_1[left_context + right_context], axis=0)
layer_2 = sigmoid(layer_1.dot(weights_1_2[target_samples].T))

# >>> Finding delta for every layer

layer_2_delta = layer_2 - layer_2_target
layer_1_delta = layer_2_delta.dot(weights_1_2[target_samples])

# >>> Updating weights

weights_0_1[left_context + right_context] -= layer_1_delta * alpha
weights_1_2[target_samples] -= np.outer(layer_2_delta, layer_1) * alpha

# >>> Interpretation

if(rev_i % 250  == 0):

    sys.stdout.write('\rProgress: ' + str(rev_i/float(len(input_dataset) * iterations)) + " " + str(similiar('terrible')))

    sys.stdout.write('\rProgress: ' + str(rev_i/float(len(input_dataset) * iterations)))

print(similiar('terrible'))























