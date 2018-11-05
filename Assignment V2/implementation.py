import tensorflow as tf
import glob
import os
from string import punctuation
import re
from tensorflow.contrib import rnn

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 128  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than','but','until','nor', '<br />', 'etc'})


def preprocess(review):
    # path='./data/train'
    # dir = os.path.dirname(__file__)
    # file_list = glob.glob(os.path.join(dir, path + '/pos/*'))
    # file_list.extend(glob.glob(os.path.join(dir, path + '/neg/*')))
    # print("Parsing %s files" % len(file_list))
    # for i, f in enumerate(file_list):
    #     with open(f, "r") as openf:
    # review = openf.read()
    after_remove = list()
    clean_sentence = list()

    review=re.sub(r'[^\w\s]','',review)
    #review = re.sub(r"[{}]+".format(punctuation), " ", review)
    for word in review.strip().split(' '):

        if word not in stop_words:
            clean_sentence.append(word.lower())
    if (len(clean_sentence) < MAX_WORDS_IN_REVIEW):
        for i in range(MAX_WORDS_IN_REVIEW - len(clean_sentence)):
            clean_sentence.append(' ')
    #after_remove.append(" ".join(clean_sentence))
   # processed_review = after_remove[0].lower()


    return clean_sentence


def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    LR = 0.001
    lstmUnits = 40
    ### input data

    input_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")

    labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 2], name="labels")
    dropout_keep_prob = tf.placeholder_with_default(0.6, shape=(), name="dropout_keep_prob")

    lstmCell = rnn.BasicLSTMCell(lstmUnits)
    lstmCell_dropout = rnn.DropoutWrapper(cell=lstmCell, input_keep_prob=dropout_keep_prob,
                                          output_keep_prob=dropout_keep_prob)
    outputs, state = tf.nn.dynamic_rnn(lstmCell_dropout, input_data, dtype=tf.float32)
    outputs = tf.transpose(outputs, [1, 0, 2])
    value = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

    weight = tf.Variable(tf.truncated_normal(shape=[lstmUnits, 2], stddev=0.2))
    bias = tf.Variable(tf.constant(0.1, shape=[2]))

    logits = tf.matmul(value, weight) + bias
    prediction = tf.nn.softmax(logits)

    correctPred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    # acc
    Accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32), name="accuracy")
    # loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=labels),
        name="loss"
    )
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss)

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss

