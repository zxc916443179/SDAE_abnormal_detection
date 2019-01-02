import time
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import utils
import numpy as np
from sklearn import svm
import pickle

tf.flags.DEFINE_string("datasetPath", './data/appearance_spliced_images/appearance.p', "dataset path")
tf.flags.DEFINE_integer("max", 10000, "max number of dataset")
tf.flags.DEFINE_string("checkpoint_dir", "none", "loading latest checkpoint")
tf.flags.DEFINE_string('label_dir', './label/label15.p', "dir of label")
flags = tf.flags.FLAGS
graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        print('loading checkpoint in dir: %s' % flags.checkpoint_dir)
        checkpoint_file = tf.train.latest_checkpoint(flags.checkpoint_dir)
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        print('load success')

        encode = graph.get_operation_by_name("hidden_layer_128/Sigmoid").outputs[0]
        recon = graph.get_operation_by_name("finetuning_decoder_3/Sigmoid").outputs[0]
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        mask = graph.get_operation_by_name("mask").outputs[0]
        # dataset = u.loadDataset(batch_size=flags.max, max=flags.max)
        label = utils.loadlabel(dir=flags.label_dir)
        dataset = utils.load_whole_dataset(max = flags.max, dataset_dir=flags.datasetPath)
        # dataset = next(dataset)
        mask_ts = np.random.binomial(1, 1, dataset.shape)
        encoder_result, recon_result = sess.run([encode, recon], feed_dict={
            input_x: dataset, mask: mask_ts})
        # n_examples = 20
        # fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
        # for example_i in range(n_examples):
        #     axs[0][example_i].imshow(
        #         # np.reshape(test_xs[example_i, :], (28, 28)))
        #         np.reshape(dataset[example_i, :], (15, 15)))
        #     axs[1][example_i].imshow(
        #         # np.reshape([recon[example_i, :] + mean_img], (28, 28)))
        #         np.reshape([recon_result[example_i, :]], (15, 15)))
        # plt.show()
        # plt.waitforbuttonpress()
        print ('Plot complete now showing...')
        accs = []
        test = utils.load_whole_dataset(max = len(label), dataset_dir="./label.appearance15.p")
        mask_ts = np.random.binomial(1, 1, test.shape)
        test_result = sess.run([encode], feed_dict={
            input_x: test, mask: mask_ts
        })
        print('starting cross fold validation')
        for i in range(0, 0.1, 0.0001):
            print('step i:{}'.format(i))
            clf = svm.OneClassSVM(kernel='rbf', gamma='auto', nu=i)
            clf.fit(encoder_result)
            pre = clf.predict(test_result)
            prediction = np.equal(pre, label)
            accuracy = np.mean(np.cast[np.float](prediction))
            print('accuracy:{:g}'.format(accuracy))
            accs.append(accuracy)
        i = np.argmax(accs)
        print('biggest nu is:{} acc:{:g}'.format(i * 0.0001, accs[i]))
        # with open("./svm.model", "rb") as f:
        #     new_svm = pickle.load(f)
        #     pre = new_svm.predict(encoder_result)
        #     plt.scatter(range(pre.shape[0]) ,pre)
        #     plt.waitforbuttonpress()
        #     prediction = np.equal(pre, label)
        #     accuracy = np.mean(np.cast[np.float](prediction))
        #     print('accuracy:{:g}'.format(accuracy))
        # clf = svm.OneClassSVM(kernel='rbf', gamma='auto', nu=1e-3)
        # clf.fit(encoder_result)
        # with open('./svm.model', 'wb') as m:
        #     pickle.dump(clf, m)