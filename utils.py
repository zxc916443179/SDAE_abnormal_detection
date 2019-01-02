"""Some useful utilities when dealing with neural nets w/ tensorflow and data process.

Parag K. Mital, Jan. 2016, zxc 2018.
"""
import tensorflow as tf
import numpy as np
import pickle
# import matplotlib.pyplot as plt

def montage_batch(images):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.

    Parameters
    ----------
    batch : numpy.ndarray
        Input array to create montage of.

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones(
        (images.shape[1] * n_plots + n_plots + 1,
         images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter, ...]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w, :] = this_img
    return m


# %%
def montage(W):
    """Draws all filters (n_input * n_output filters) as a
    montage image separated by 1 pixel borders.

    Parameters
    ----------
    W : numpy.ndarray
        Input array to create montage of.

    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
    n_plots = int(np.ceil(np.sqrt(W.shape[-1])))
    m = np.ones(
        (W.shape[0] * n_plots + n_plots + 1,
         W.shape[1] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < W.shape[-1]:
                m[1 + i + i * W.shape[0]:1 + i + (i + 1) * W.shape[0],
                  1 + j + j * W.shape[1]:1 + j + (j + 1) * W.shape[1]] = (
                    np.squeeze(W[:, :, :, this_filter]))
    return m




# %%
def corrupt(x, mask=None):
    """Take an input tensor and add uniform masking.

    Parameters
    ----------
    x : Tensor/Placeholder
        Input to corrupt.
    mask: none
    Returns
    -------
    x_corrupted : Tensor
        50 pct of values corrupted.
    """
    import copy
    cor = copy.deepcopy(x)
    if mask is not None:
        for i in range(x.shape[0]):
            cor[i] *= mask[i]
        shuffle_indices = np.random.permutation(np.arange(cor.shape[0]))
        shuffle_data = cor[shuffle_indices]
        x = x[shuffle_indices]
        return shuffle_data, x
    n = np.random.normal(0, 0.1, (x.shape[0], x.shape[1]))
    # y = x + n
    # fig, axs = plt.subplots(2, n_examples, figsize=(10, 2))
    # for example_i in range(n_examples):
    #     axs[0][example_i].imshow(
    #         # np.reshape(test_xs[example_i, :], (28, 28)))
    #         np.reshape(y[example_i, :], (15, 15)))
    #     axs[1][example_i].imshow(
    #         # np.reshape([recon[example_i, :] + mean_img], (28, 28)))
    #         np.reshape([x[example_i, :]], (15, 15)))
    # print ('Plot complete now showing...')
    # fig.show()
    # plt.draw()
    # plt.title("1st function - dataset ones but used our dataset")
    # plt.waitforbuttonpress()
    return x + n
    # return tf.multiply(x, tf.cast(tf.random_uniform(shape=tf.shape(x),
    #                                            minval=0,
    #                                            maxval=2,
    #                                            dtype=tf.int32), tf.float32))


# %%
def weight_variable(shape):
    '''Helper function to create a weight variable initialized with
    a normal distribution

    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


# %%
def bias_variable(shape):
    '''Helper function to create a bias variable initialized with
    a constant value.

    Parameters
    ----------
    shape : list
        Size of weight variable
    '''
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)
# %%
def loadDataset(batch_size=1000, max = 0, dataset_dir=None):
    """
    load pickle dataset by batches(deprecated)

    Author
    ------
        zxc

    Parameters
    ---------- 
        batch_size: size of batch
        max: max num of dataset to load. if max == 0, load all(recommend to use load_whole_dataset bellow)
        dataset_dir: dir to dataset
    """
    opendataset = open(dataset_dir,'rb')
    dataset = pickle.load(opendataset)
    dataset = [dataset]
    i = 1
    total = 1
    while True:
        try:
            tmp = pickle.load(opendataset)
            dataset.append(tmp)
            if max != 0 and total == max:
                dataset = np.asarray(dataset)
                yield dataset
                break
            if i == batch_size:
                dataset = np.asarray(dataset)
                yield dataset
                i = 1
                dataset = [pickle.load(opendataset)]
            i += 1
            total += 1
        except Exception as e:
            print(e)
            dataset = np.asarray(dataset)
            yield dataset
            opendataset.close()
            break
    opendataset.close()

def load_whole_dataset(max=0, dataset_dir=None):
    """
    load whole pickle dataset 

    Author
    ------
        zxc

    Parameters
    ---------- 
        max: max num of dataset to load. if max == 0, load all(assume that the size of dataset is no bigger than 5000000, sure you can modify it but mind memory size)
        dataset_dir: dir to dataset
    """
    if max == 0:
        dataset = np.zeros([5000000, 225])
    else:
        dataset = np.zeros([max, 225])
    opendataset = open(dataset_dir,'rb')
    end = 0
    for i in range(dataset.shape[0]):
        try:
            dataset[i] = pickle.load(opendataset)
            end += 1
        except:
            end -= 1
            break
    print(end)
    return dataset[:end]
# %%
def loadlabel(dir=None):
    if dir is None:
        print('dir is expected type of string but get None')
    label_file = open(dir, 'rb')
    label = []
    while True:
        try:
            t = pickle.load(label_file)
            if t is 0:
                label.append(1)
            else:
                label.append(-1)
        except Exception as e:
            print(e)
            break
    return np.asarray(label)
