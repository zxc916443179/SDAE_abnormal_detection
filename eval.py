import time
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import denoising_autoencoder as u
import numpy as np
from sklearn import svm
import pickle

flags = tf.flags.FLAGS
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        checkpoint_file = tf.train.latest_checkpoint(flags.checkpoint_dir)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        encode = graph.get_operation_by_name("hidden_layer_128/Sigmoid").outputs[0]
        recon = graph.get_operation_by_name("fn_score/score").outputs[0]
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        mask = graph.get_operation_by_name("mask").outputs[0]
        dataset = u.loadDataset(batch_size=flags.max, max=flags.max)
        dataset = next(dataset)
        mask_ts = np.random.binomial(1, 1, dataset.shape)
        # batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
        encoder_result, recon_result = sess.run([encode, recon], feed_dict={
            input_x: dataset, mask: mask_ts})
        n_examples = 15
        fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
        for example_i in range(n_examples):
            axs[0][example_i].imshow(
                # np.reshape(test_xs[example_i, :], (28, 28)))
                np.reshape(dataset[example_i, :], (15, 15)))
            axs[1][example_i].imshow(
                # np.reshape([recon[example_i, :] + mean_img], (28, 28)))
                np.reshape([recon_result[example_i, :]], (15, 15)))
        print ('Plot complete now showing...')
        with open("./svm.model", "rb") as f:
            new_svm = pickle.load(f)
            # print(encoder_result.shape)
            pre = new_svm.decision_function(encoder_result).ravel() * -1
            plt.plot(range(1000), pre[:1000])
            plt.show()
            # plt.waitforbuttonpress()