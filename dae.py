import tensorflow as tf
import numpy as np
import time
import os
import datetime
import pickle
from sklearn import svm
from matplotlib import pyplot as plt
import utils
# hyper params define   
tf.flags.DEFINE_float("validation", 0.8, "ratio of train/test")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_string("datasetPath", './data/appearance_spliced_images/appearance.p', "dataset path")
tf.flags.DEFINE_integer("num_epoch", 10, "number of epoch(default: 10)")
tf.flags.DEFINE_integer("batch_size", 10, "batch size(default: 10)")
tf.flags.DEFINE_integer("max", 10000, "max number of dataset")
tf.flags.DEFINE_string("checkpoint_dir", "none", "loading latest checkpoint")
tf.flags.DEFINE_float("corrupt_prob", 0.0003, "corrupt data ratio")
tf.flags.DEFINE_string("dimensions", "1024,512,256,128", "dimensions of hidden layers (default:[1024, 512, 256, 128]")
tf.flags.DEFINE_float("momentum", 0.9, "learning momentum(default:0.9)")

flags = tf.flags.FLAGS

class DAE(object):
    def __init__(self, hidden_layers=[1024, 512, 256, 128], l2_reg_lambda = 0.0, momentum=0.9):
        self.input_x = tf.placeholder(tf.float32, [None, 225], name="input_x")
        self.mask = tf.placeholder(tf.float32, [None, 225], name="mask")
        self.dimensions = hidden_layers
        self.Y = self.input_x * self.mask
        self.momentum = momentum
        n_input = 225
        current_input = self.input_x
        self.Ws = []
        self.scores = []
        self.params = []
        self.layer_output = [] # normal output
        self.da_output = [] # da output
        self.l2_losses = [tf.constant(0.0) for _ in hidden_layers]
        for layer_i, dimension in enumerate(hidden_layers):
            if layer_i == 0:
                current_input = self.input_x
            else:
                current_input = self.layer_output[layer_i - 1]
            with tf.name_scope("hidden_layer_%d" % dimension):
                W = tf.Variable(tf.random_uniform([n_input, dimension], -1.0, 1.0), name="W")  
                b = tf.Variable(tf.constant(0.0, shape=[dimension]), name="b")
                self.Ws.append(W)
                output = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
                self.layer_output.append(output)
                self.l2_losses[layer_i] += tf.nn.l2_loss(W)
                self.params.extend([W, b])
            with tf.name_scope("decoder_%d" % dimension):
                W_prime = tf.transpose(self.Ws[layer_i], name="W")
                b_prime = tf.Variable(tf.constant(0.1, shape=[n_input]), name="b")
                self.y = tf.nn.sigmoid(tf.matmul(output, W_prime) + b_prime)
                self.l2_losses[layer_i] += tf.nn.l2_loss(W_prime)
            with tf.name_scope("score_%d" % dimension):
                loss = tf.pow(self.y - current_input, 2)
                score = tf.reduce_sum(loss, name="score_%d" % dimension) + self.l2_losses[layer_i] * l2_reg_lambda
                self.scores.append(score)
                self.da_output.append(self.y)
            n_input = dimension



    def pretrain(self, batch_size, num_epoch, sess):
        graph = tf.Graph()
        # init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        sess.run(tf.global_variables_initializer())
        # sess.run(init)
        for i in range(len(self.dimensions)):

            # learning_rate = 0.01
            
            global_step = tf.Variable(0, trainable=False, name="global_step")
            learning_rate = 0.001
            # learning_rate = tf.train.exponential_decay(0.01, global_step, flags.max / flags.batch_size, 0.98, staircase=True)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=self.momentum)
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            grads_and_vars = optimizer.compute_gradients(self.scores[i])
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            sess.run(tf.initialize_all_variables())
            #define summaries
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)
            score_summary = tf.summary.scalar("score", self.scores[i])

            train_summary_op = tf.summary.merge([score_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


            for j in range(num_epoch):
                for batch in utils.loadDataset(batch_size, max=flags.max, dataset_dir=flags.datasetPath):
                    mask_t = np.random.binomial(1, 1 - flags.corrupt_prob, batch.shape)
                    self.batch = batch
                    # mean_img = np.mean(batch, axis=1)
                    # batch = np.array([img - mean_img for img in batch.T])
                    # batch = batch.T
                    _, score, step, summaries, self.recon = sess.run([train_op, self.scores[i], global_step, train_summary_op, self.da_output[i]], feed_dict={
                        self.input_x: batch, self.mask: mask_t
                    })
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % 100 == 0:
                        print("traning Layer_%d " % i + "epoch:%d " % j + "step: %d" % step + "  score: {}".format(score))
                    train_summary_writer.add_summary(summaries, step)
                    if current_step % 1000 == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

        self.finetuning(batch_size, num_epoch, sess, saver, out_dir)

    def finetuning(self, batch_size, num_epoch, sess, saver, out_dir):

        current_input = self.layer_output[len(self.dimensions) - 1]
        self.ft_losses = [tf.constant(0.0) for _ in self.dimensions]
        for layer_i, dimension in enumerate(self.dimensions):
            print(2 - layer_i)
            if layer_i == 3:
                n_output = 225
            else:
                n_output = self.dimensions[2 - layer_i]
                print(n_output)
            with tf.name_scope("finetuning_decoder_%i" % layer_i):
                W = tf.transpose(self.Ws[3 - layer_i], name="W")
                b = tf.Variable(tf.constant(0.1, shape=[n_output]), name="b")
                self.out_put = tf.nn.sigmoid(tf.matmul(current_input, W) + b)
                self.ft_losses[layer_i] += tf.nn.l2_loss(W)
                current_input = self.out_put
        with tf.name_scope('fn_score'):
            loss = tf.pow(self.out_put - self.input_x, 2)
            self.score = tf.reduce_sum(loss, name="score") + self.ft_losses[layer_i] * flags.l2_reg_lambda
        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = 0.001
        # learning_rate = tf.train.exponential_decay(0.01, global_step, flags.max / flags.batch_size, 0.98, staircase=True)
        saver = tf.train.Saver(tf.global_variables())
        # optimizer = tf.train.RMSPropOptimizer(learning_rate, momentum=self.momentum)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.score)
        finetune_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        sess.run(tf.initialize_all_variables())
        timestamp = str(int(time.time()))
        # out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        score_summary = tf.summary.scalar("score", self.score)

        finetune_summary_op = tf.summary.merge([score_summary, grad_summaries_merged])
        finetune_summary_dir = os.path.join(out_dir, "summaries", "finetune")
        finetune_summary_writer = tf.summary.FileWriter(finetune_summary_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        print("Starting finetuning")
        for j in range(num_epoch):
            for batch in loadDataset(batch_size, max=flags.max):
                mask_t = np.random.binomial(1, 1 - flags.corrupt_prob, batch.shape)
                # mean_img = np.mean(batch, axis=1)
                # batch = np.array([img - mean_img for img in batch.T])
                # batch = batch.T
                _, score, step, summaries, recon = sess.run([finetune_op, self.score, global_step, finetune_summary_op, self.out_put], feed_dict={
                    self.input_x: batch, self.mask: mask_t
                })
                current_step = tf.train.global_step(sess, global_step)
                if current_step % 100 == 0:
                    print("finetuning  step: %d" % step + "  score: {}".format(score))
                finetune_summary_writer.add_summary(summaries, step)
                if current_step % 1000 == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
        # with graph.as_default():
        #     with sess.as_default():
        n_examples = 15
        test_xs = next(loadDataset(256, 256))
        mask = np.random.binomial(1, 1, test_xs.shape)
        score, recon, encodes = sess.run([self.score, self.out_put, self.layer_output], feed_dict={
            self.input_x: test_xs, self.mask: mask
        })
        # fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
        # for example_i in range(n_examples):
        #     axs[0][example_i].imshow(
        #         # np.reshape(test_xs[example_i, :], (28, 28)))
        #         np.reshape(test_xs[example_i, :], (15, 15)))
        #     axs[1][example_i].imshow(
        #         # np.reshape([recon[example_i, :] + mean_img], (28, 28)))
        #         np.reshape([recon[example_i, :]], (15, 15)))
        # print ('Plot complete now showing...')
        clf = svm.OneClassSVM(kernel='rbf', gamma='auto', nu=1e-3)
        clf.fit(encodes[3])
        with open('./svm.model', 'wb') as m:
            pickle.dump(clf, m)
        # fig.show()
        # plt.draw()
        # plt.title("1st function - dataset ones but used our dataset")
        # plt.waitforbuttonpress()


if __name__ == "__main__":
    dimensions = list(map(int, flags.dimensions.split(",")))
    da = DAE(dimensions, l2_reg_lambda=flags.l2_reg_lambda, momentum=flags.momentum)
    sess = tf.Session()
    da.pretrain(flags.batch_size, flags.num_epoch, sess)
    # da.finetuning(flags.batch_size, flags.num_epoch, flags.checkpoint_dir)