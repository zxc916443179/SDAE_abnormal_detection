import sys
import math
import pickle
import numpy as np
import tensorflow as tf
from utils import corrupt
import argparse
import os
import time
import datetime
from sklearn import svm
# from libs.utils import corrupt

listOfDatasets = ['appearance_features_train.p','appearance_features_test.p','motion_features_train.p','motion_features_original_test.p']

# Provide data-set path here
tf.flags.DEFINE_float("validation", 0.8, "ratio of train/test")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_string("datasetPath", './data/appearance_spliced_images/appearance.p', "dataset path")
tf.flags.DEFINE_integer("num_epoch", 10, "number of epoch(default: 10)")
tf.flags.DEFINE_integer("batch_size", 10, "batch size(default: 10)")
tf.flags.DEFINE_integer("max", 10000, "max number of dataset")
tf.flags.DEFINE_string("model_path", "none", "loading model path(default: none(don't load))")
tf.flags.DEFINE_string("checkpoint_dir", "none", "loading latest checkpoint")
tf.flags.DEFINE_float("corrupt_prob", 0.003, "corrupt data ratio")
tf.flags.DEFINE_string("dimensions", "1024,512,256,128", "dimensions of hidden layers (default:[1024, 512, 256, 128]")
flags = tf.flags.FLAGS
# arg = argparse.ArgumentParser()
# arg.add_argument("-p", "--path", help="dataset path")
# arg.add_argument("-v", "--validation", help="validation ratio")
# args = vars(arg.parse_args())
# if args['path'] is None:
#     datasetPath = os.path.join('./data/appearance_spliced_images/', 'appearance.p')
# else:
#     datasetPath = args['path']
#datasetPath = 'appearancedataset.p'
def loadDataset(batch_size=1000, max = 0):
    opendataset = open(flags.datasetPath,'rb')
    # dataset = pickle.load(opendataset)
    dataset = np.zeros([1, 225])
    dataset[0] = pickle.load(opendataset)
    i = 1
    total = 1
    while True:
        try:
            tmp = pickle.load(opendataset)
            dataset = np.append(dataset, [tmp], axis=0)
            if i % 100 is 0:
                pass
            # print(dataset)
            if max != 0 and total == max:
                print('return')
                dataset = np.asarray(dataset)
                yield dataset
                break
            if i == batch_size:
                dataset = np.asarray(dataset)
                yield dataset
                i = 1
                dataset = np.zeros([1, 225])
                dataset[0] = pickle.load(opendataset)
            i += 1
            total += 1
        except Exception as e:
            print(e)
            dataset = np.asarray(dataset)
            yield dataset
            opendataset.close()
            break
    opendataset.close()

# opendataset = open(datasetPath,'r')
# dataset = pickle.load(opendataset)
# opendataset.close()
def partition(dataset, ratio=0.8, shuffle=False):
    start_index = 0
    end_index = np.int(np.round(dataset.shape[0] * ratio))
    dataset_train = dataset[start_index: end_index, :]
    dataset_test = dataset[end_index: dataset.shape[0], :]
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(dataset_train.shape[0]))
        shuffle_data = dataset_train[shuffle_indices]
    else:
        shuffle_data = dataset_train
    return shuffle_data, dataset_test


# if sys.version_info.major == 3:
#     print (dataset[:, 0:500].shape)
#     print (dataset[:, 501:700].shape)
# else:
#     print (dataset[:, 0:500].shape)
#     print (dataset[:, 501:700].shape)


# %%
#def autoencoder(dimensions=[784, 512, 256, 64]):
class autoencoder(object):

    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    """
    # input to the network
    def __init__(self, dimensions=[225, 128, 64, 32, 16], l2_reg_lambda = 0.0):
        self.x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
        # print(tf.shape(self.x))
        # Probability that we will corrupt input.
        # This is the essence of the denoising autoencoder, and is pretty
        # basic.  We'll feed forward a noisy input, allowing our network
        # to generalize better, possibly, to occlusions of what we're
        # really interested in.  But to measure accuracy, we'll still
        # enforce a training signal which measures the original image's
        # reconstruction cost.
        #
        # We'll change this to 1 during training
        # but when we're ready for testing/production ready environments,
        # we'll put it back to 0.
        self.x_ = tf.placeholder(tf.float32, [None, dimensions[0]], name="x_")
        l2_loss = tf.constant(0.0)
        current_input = self.x_
        # Build the encoder
        self.encoder = []
        for layer_i, n_output in enumerate(dimensions[1:]):
            with tf.name_scope("encoder_layer_%s" % layer_i):
                print("Layer : " + str(layer_i))
                n_input = int(current_input.get_shape()[1])
                W = tf.Variable(
                    tf.random_uniform([n_input, n_output], -1.0, 1.0), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[n_output]), name="b")
                self.encoder.append(W)
                # output = tf.nn.tanh(tf.matmul(current_input, W) + b)
                ae_output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                current_input = ae_output
        # latent representation
        self.z = ae_output
        # Here use the classifier for the latent representaion
        
        self.encoder.reverse()
        # Build the decoder using the same weights
        for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
            with tf.name_scope("decoder_layer_%s" % layer_i):
                n_input = int(current_input.get_shape()[1])
                # W = tf.Variable(tf.random_uniform([n_input, n_output], 1.0, 10), name="W")
                W = tf.transpose(self.encoder[layer_i], name="W")
                b = tf.Variable(tf.constant(0.1, shape=[n_output]), name="b")   
                # output = tf.nn.tanh(tf.matmul(current_input, W) + b)
                self.output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                current_input = self.output
        with tf.name_scope('score'):  
            loss = tf.pow(self.output - self.x, 2)
            # loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.x)
            self.score = tf.reduce_sum(loss, name="score") + l2_loss * l2_reg_lambda

# %%


def test_dataset():
    import tensorflow as tf
    import matplotlib.pyplot as plt

    # %%
    # load Dataset

    # dataset = dataset ??????# Here we will set out dataset 
    # dataset_train, dataset_test = dataset[:,0:35], dataset[:,36:51]
    # print ("Train slice of dataset" + str(dataset_train.shape))
    # print ("Test slice of dataset" + str(dataset_test.shape))
    # mean_img = np.mean(dataset_train, axis=0)
    # print ("Mean Image : "+str(mean_img.shape))
    with tf.Graph().as_default():
        ae = autoencoder(l2_reg_lambda=flags.l2_reg_lambda)

        # %%
        # learning_rate = 1e-5
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.train.exponential_decay(0.0001, global_step, flags.max/flags.batch_size, 0.96, staircase=True)
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(ae.score)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        # train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(ae.score, global_step=global_step)

        # %%
        # We create a session to use the graph
        sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        input_hist_summary = tf.summary.histogram("input/grad/hist", ae.x)
        output_hist_summary = tf.summary.histogram("output/grad/hist", ae.output)
        grad_summaries.append(input_hist_summary)
        grad_summaries.append(output_hist_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)


        score_summary = tf.summary.scalar("score", ae.score)
        output_summary = tf.summary.tensor_summary("output", ae.output)

        train_summary_op = tf.summary.merge([score_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


        dev_summary_op = tf.summary.merge([score_summary, output_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if flags.model_path is not "none":
            print('loading model')
            saver = tf.train.import_meta_graph(flags.model_path)
            saver.restore(sess, tf.train.latest_checkpoint(flags.checkpoint_dir))
            print('loading success')
        else:
            init = tf.global_variables_initializer()
            sess.run(init)

        test_xs = np.zeros([1, 225])
        # test_xs = []
        # test_xs = np.asarray(test_xs)
        # %%
        # Fit all training data
        def train_step(test_xs):
            batch_size = flags.batch_size
            # batch_size = 50
            n_epochs = flags.num_epoch
            mask = np.random.binomial(1, 1 - flags.corrupt_prob, (int(np.round(batch_size * flags.validation)) + 1, 225))
            # print(mask[:5])
            for epoch_i in range(n_epochs):
                # print dataset_train.shape[1] // batch_size
                datasets = loadDataset(batch_size=batch_size, max=flags.max)
                f = 0
                for dataset in datasets:
                    dataset_train, dataset_test = partition(dataset, shuffle=False)
                    mean_img = np.mean(dataset_train, axis=1)
                    dataset_train = np.array([img - mean_img for img in dataset_train.T])
                    dataset_train = dataset_train.T
                    dataset_train_, dataset_train = corrupt(dataset_train, mask=mask)
                    _, score, step, summaries = sess.run([train_op, ae.score, global_step, train_summary_op], feed_dict={
                        ae.x: dataset_train, ae.x_: dataset_train_})
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % 100 == 0:
                        print("epoch:{} step:{} score:{}".format(epoch_i, step, score))
                    train_summary_writer.add_summary(summaries, step)
                    if current_step % 1000 == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))
                # score, step, summaries, output, W= sess.run([ae.score, global_step, dev_summary_op, ae.output, ae.encoder], feed_dict={
                #         ae.x: test_xs,
                #         ae.x_: test_xs})
                # print("evaluation:\nscore:{}".format(score))
            test_xs = np.asarray(test_xs)
            print ("Testxs : " +str(test_xs.shape))
            return test_xs
            # test_xs_norm = np.array([img - mean_img for img in test_xs])
            # print ("Test xs Norm : " + str(test_xs_norm.shape))
        test_xs = train_step(test_xs)
        test_xs = loadDataset(batch_size=1000, max=1000)
        test_xs = next(test_xs)
        # %%
        # Plot example reconstructions
        n_examples = 30
        # test_xs, _ = datasetest.next_batch(n_examples)
        # test_xs = dataset_test[batch_i:batch_i + batch_size, :]
        recon, score, encode = sess.run([ae.output, ae.score, ae.z], feed_dict={
            ae.x: test_xs, ae.x_: test_xs})
        print ("Reconstruction shape: " + str(recon.shape))
        print ("Reconstruction Complete")
        print('recon:{}'.format(recon[0:5]))
        print('test:{}'.format(test_xs[0:5]))
        print('score:{}'.format(score))
        recon_out = open("./recon.out", "wb")
        recon_out.writelines(recon)
        recon_out.close()
        fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(
                # np.reshape(test_xs[example_i, :], (28, 28)))
                np.reshape(test_xs[example_i, :], (15, 15)))
            axs[1][example_i].imshow(
                # np.reshape([recon[example_i, :] + mean_img], (28, 28)))
                np.reshape([recon[example_i, :]], (15, 15)))
        print ('Plot complete now showing...')
        clf = svm.OneClassSVM(kernel='rbf', gamma='auto', nu=0.5)
        clf.fit(encode)
        with open('./svm.model', 'wb') as m:
            pickle.dump(clf, m)
        fig.show()
        plt.draw()
        plt.title("1st function - dataset ones but used our dataset")
        plt.waitforbuttonpress()


def train_appearance_features():
    import tensorflow as tf
    import matplotlib.pyplot as plt

    # %%
    # load Dataset

    appearance_dataset = dataset # Here we will set out dataset
    mean_img = np.mean(appearance_dataset)
    appearance_train, appearance_test = dataset[:,0:35], dataset[:,36:51]
    print (appearance_train.shape)
    print (appearance_test.shape)
    # mean_img = np.mean(dataset.train.images, axis=0)
    ae = autoencoder(dimensions=[225, 1024, 512, 256, 64])

    # %%
    learning_rate = 0.001
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    batch_size = 6
    # batch_size = 50
    n_epochs = 2
    for epoch_i in range(n_epochs):
        # print dataset_train.shape[1] // batch_size
        for batch_i in range(appearance_train.shape[1] // batch_size):
            batch_xs = appearance_train.T[batch_i:batch_i + batch_size,:]
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={
                ae['x']: train, ae['corrupt_prob']: [1.0]})
        print(epoch_i, sess.run(ae['cost'], feed_dict={
            ae['x']: train, ae['corrupt_prob']: [1.0]}))

    # %%
    # Plot example reconstructions
    n_examples = 15
    # test_xs, _ = dataset.test.next_batch(n_examples)
    for batch_i in range(appearance_train.shape[1]//batch_size):
        print (batch_i, appearance_train.shape[1],batch_size)
        test_xs = appearance_test.T[batch_i:batch_i+batch_size,:]
        test_xs_norm = np.array([img - mean_img for img in test_xs])
        recon = sess.run(ae['y'], feed_dict={
        ae['x']: test_xs_norm, ae['corrupt_prob']: [0.0]})
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            # np.reshape(test_xs[example_i, :], (28, 28)))
            np.reshape(test_xs[example_i, :], (15, 15)))
        axs[1][example_i].imshow(
            np.reshape([recon[example_i, :] + mean_img], (15, 15)))
    fig.show()
    plt.draw()
    plt.title('Appearance features')
    plt.waitforbuttonpress()

def train_motion_features():
    pass

def train_joint_features():
    # type: () -> object
    pass

if __name__ == '__main__':
    test_dataset()
    # train_appearance_features()
    # train_motion_features()
    # train_joint_features()
