import time
import os
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import denoising_autoencoder as u
import numpy as np
from sklearn import svm
import pickle
# # from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
# tf.flags.DEFINE_float("validation", 0.8, "ratio of train/test")
# tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
# tf.flags.DEFINE_string("datasetPath", './data/appearance_spliced_images/appearance.p', "dataset path")
# tf.flags.DEFINE_integer("num_epoch", 10, "number of epoch(default: 10)")
# tf.flags.DEFINE_integer("batch_size", 10, "batch size(default: 10)")
# tf.flags.DEFINE_integer("max", 10000, "max number of dataset")
# tf.flags.DEFINE_string("model_path", "none", "loading model path(default: none(don't load))")
# tf.flags.DEFINE_string("checkpoint_dir", "none", "loading latest checkpoint")
# tf.flags.DEFINE_float("corrupt_prob", 0.3, "corrupt data ratio")
flags = tf.flags.FLAGS
# learning_rate = 0.01
training_epochs = flags.num_epoch
batch_size = flags.batch_size
print(flags.batch_size)
print(flags.corrupt_prob)
n_input = 225
# class DAE (object):
    # def __init__(self):
X_ = tf.placeholder("float", [None, n_input], name="input_x")
# X = tf.placeholder("float", [None, n_input])
mask = tf.placeholder("float", [None, n_input], name='mask')
X = X_ * mask
n_hidden_1 = 64
n_hidden_2 = 32
n_hidden_3 = 16
n_hidden_4 = 8
weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],)),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b4': tf.Variable(tf.random_normal([n_input])),
}
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                biases['encoder_b3']))
    # 为了便于编码层的输出，编码层随后一层不使用激活函数
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                    biases['encoder_b4'], name="fn_score")
    return layer_4

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']), name="output")
    return layer_4

encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

y_pred = decoder_op
y_true = X_
cost = tf.reduce_sum(tf.pow(y_true - y_pred, 2), name="cost")

g = tf.Graph()
g.as_default()
with tf.Session() as sess:
    # tf.initialize_all_variables() no long valid from
    # 2017-03-02 if using tensorflow >= 0.12
    # learning rate dynamic
    # if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    #     init = tf.initialize_all_variables()
    # else:
    #     init = tf.global_variables_initializer()
    # learning_rate = 0.001
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(1e-3, global_step, flags.max / batch_size, 0.98, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars = optimizer.compute_gradients(cost)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    # 为什么与优化器也有关系
    # train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost, global_step=global_step)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp + "batch_size_{}_epoch_{}".format(flags.batch_size, flags.num_epoch)))
    print("Writing to {}\n".format(out_dir))
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    input_hist_summary = tf.summary.histogram("input/hist", X)
    output_hist_summary = tf.summary.histogram("output/hist", y_pred)
    grad_summaries.append(input_hist_summary)
    grad_summaries.append(output_hist_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    score_summary = tf.summary.scalar("score", cost)


    train_summary_op = tf.summary.merge([score_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)


    dev_summary_op = tf.summary.merge([score_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if flags.model_path is not "none":
        saver = tf.train.import_meta_graph(flags.model_path)
        saver.restore(sess, tf.train.latest_checkpoint(flags.checkpoint_dir))
    else:
        sess.run(tf.global_variables_initializer())
    # sess.run(tf.global_variables_initializer())
    # total_batch = int(mnist.train.num_examples/batch_size)
    dataset = u.loadDataset(batch_size=flags.max, max=flags.max)
    dataset = next(dataset)
    dataset_train, dataset_test = u.partition(dataset, flags.validation, True)
    total_batch = int(dataset_train.shape[0] / batch_size)
    mask_ts = np.random.binomial(1, 1, dataset_test.shape)
    for epoch in range(training_epochs):
        for i in range(total_batch):
            corruption_level = flags.corrupt_prob
            batch_xs = dataset_train[i * batch_size: i * batch_size + batch_size]
            mean_img = np.mean(batch_xs, axis=1)
            batch_xs = np.array([img - mean_img for img in batch_xs.T])
            batch_xs = batch_xs.T
            mask_np = np.random.binomial(1, 1 - corruption_level, batch_xs.shape)
            # batch_xs, batch_ys = mnist.train.next_batch(batch_size)  # max(x) = 1, min(x) = 0
            _, c, step, summaries= sess.run([train_op, cost, global_step, train_summary_op], feed_dict={X_: batch_xs, mask: mask_np})
            if step % 100 == 0:
                print("Epoch:", '%04d' % (epoch+1), "step:%d " % step,"cost=", "{:.9f}".format(c))
                train_summary_writer.add_summary(summaries, step)
            if step % 1000 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=step)
        print("Evaluation:\n")
        _, c, summaries = sess.run([train_op, cost, dev_summary_op], feed_dict={
            X_: dataset_test, mask: mask_ts
        })
        print("cost:{}".format(c))
        dev_summary_writer.add_summary(summaries, global_step=tf.train.global_step(sess, global_step))
        print ('Plot complete now showing...')
        
    print("Optimization Finished!")
    # dataset = u.loadDataset(batch_size=12000, max=12000)
    # val = dataset[10000: 12000]
    print(dataset_test[0:5])
    encoder_result, recon = sess.run([encoder_op, y_pred], feed_dict={
        X_: dataset_test, mask: mask_ts})
    print(recon[:5])
    print("cost:{}".format(cost))
    out = open('./out', 'wb')
    pickle.dump(recon, out)
    out.close()
    n_examples = 15
    fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            # np.reshape(test_xs[example_i, :], (28, 28)))
            np.reshape(dataset_test[example_i, :], (15, 15)))
        axs[1][example_i].imshow(
            # np.reshape([recon[example_i, :] + mean_img], (28, 28)))
            np.reshape(recon[example_i, :], (15, 15)))
    print ('Plot complete now showing...')
    clf = svm.OneClassSVM(kernel='rbf', gamma='auto', nu=0.5)
    clf.fit(encoder_result)
    with open('./svm.model', 'wb') as m:
        pickle.dump(clf, m)
    fig.show()
    plt.draw()
    plt.title("1st function - dataset ones but used our dataset")
    plt.waitforbuttonpress()
    # plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    # plt.colorbar()
    # plt.show()

    # if __name__ == '__main__':
    # train()
    # train_appearance_features()
    # train_motion_features()
    # train_joint_features()
