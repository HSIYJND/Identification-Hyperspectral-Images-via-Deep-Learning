
from tensorflow.contrib import rnn
from __future__ import print_function
from PIL import Image
import tensorflow as tf
import numpy as np
import time
import os

time_steps = 28
num_units = 128
n_input = 28
learning_rate = 0.001
n_classes = 2
batch_size= 128
img_size = 28
save_step = 500

def get_image_paths(img_dir):
    filenames = os.listdir(img_dir)
    filenames = [os.path.join(img_dir, item) for item in filenames]
    return filenames

pos_filenames = get_image_paths("./101-150/dip")
neg_filenames = get_image_paths("./101-150/hap")
print("num of dip samples is %d" % len(pos_filenames))
print("num of hap samples is %d" % len(neg_filenames))

TRAIN_SEC, TEST_SEC = 0.8, 0.2
pos_sample_num = len(pos_filenames)
neg_sample_num = len(neg_filenames)
np.random.shuffle(np.arange(len(pos_filenames)))
np.random.shuffle(np.arange(len(neg_filenames)))
pos_train = pos_filenames[: int(pos_sample_num * TRAIN_SEC)]
pos_test = pos_filenames[int(pos_sample_num * TRAIN_SEC) :]
neg_train = neg_filenames[: int(neg_sample_num * TRAIN_SEC)]
neg_test = neg_filenames[int(neg_sample_num * TRAIN_SEC) :]

print("dip sample : train num is %d, test num is %d" % (len(pos_train), len(pos_test)))
print("hap sample : train num is %d, test num is %d" % (len(neg_train), len(neg_test)))

all_train, all_test = [], []
all_train_label, all_test_label = [], []
all_train.extend(pos_train)
all_train.extend(neg_train)
all_test.extend(pos_test)
all_test.extend(neg_test)
pos_train_label, pos_test_label = np.ones(len(pos_train), dtype=np.int64), np.ones(len(pos_test), dtype=np.int64)
neg_train_label, neg_test_label = np.zeros(len(neg_train), dtype=np.int64), np.zeros(len(neg_test), dtype=np.int64)
all_train_label = np.hstack((pos_train_label, neg_train_label))
all_test_label = np.hstack((pos_test_label, neg_test_label))
print("train num is %d, test num is %d" % (len(all_train), len(all_test)))
print("train_label num is %d, test_label num is %d" % (len(all_train_label), len(all_test_label)))

def resize_img(img_path, shape):
    im = Image.open(img_path)
    im = im.resize(shape)
    im = im.convert('L')
    return im

def save_as_tfrecord(samples, labels, bin_path):
    assert len(samples) == len(labels)
    writer = tf.python_io.TFRecordWriter(bin_path)
    img_label = list(zip(samples, labels))
    np.random.shuffle(img_label)
    for img, label in img_label:
        im = resize_img(img, (img_size, img_size))
        im_raw = im.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[im_raw]))
        }))
        writer.write(example.SerializeToString())
    writer.close()
    
save_as_tfrecord(all_train, all_train_label, "train.tfrecord")
save_as_tfrecord(all_test, all_test_label, "test.tfrecord")

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [img_size, img_size, 1])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 # normalize
    label = tf.cast(features['label'], tf.int32)
    label = tf.sparse_to_dense(label, [2], 1, 0)
    return img, label

def input_pipeline(filenames, batch_size, num_epochs=None):
    filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_and_decode(filename_queue)
    min_after_dequeue = 1000
    num_threads = 4
    capacity = min_after_dequeue + (num_threads + 3) * batch_size
    example_batch, label_batch = tf.train.shuffle_batch(
        [example, label], batch_size=batch_size, capacity=capacity, num_threads = num_threads,
        min_after_dequeue=min_after_dequeue)
    return example_batch, label_batch

img_batch, label_batch = input_pipeline(["train.tfrecord"], batch_size)
test_img_batch, test_label_batch = input_pipeline(["test.tfrecord"], batch_size)

out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
out_bias = tf.Variable(tf.random_normal([n_classes]))

x = tf.placeholder(tf.float32, [None,time_steps, n_input])
y_ = tf.placeholder(tf.int32, [None, n_classes])
input = tf.unstack(x, time_steps, 1)

lstm_layer = rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs, _ = rnn.static_rnn(lstm_layer,input, dtype="float32")
prediction = tf.matmul(outputs[-1], out_weights) + out_bias

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()
with tf.Session() as sess:
    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    threads = tf.train.start_queue_runners(coord=coord)
    iter=1
    try:
        while not coord.should_stop() and iter < 5000:
            iter += 1
            imgs, labels = sess.run([img_batch, label_batch])
            imgs = imgs.reshape((batch_size,time_steps,n_input))
            sess.run(opt, feed_dict = {x : imgs, y_ : labels})
            if iter % 20 == 0:
                acc = sess.run(accuracy, feed_dict = {x : imgs, y_ : labels})
                los = sess.run(loss, feed_dict = {x : imgs, y_ : labels})
                print("For iter ", iter)
                print("Accuracy", acc)
                print("Loss", los)
                print("__________________")
            if iter % save_step == 0:  
                save_path = saver.save(sess, 'graph.ckpt', global_step=iter)
                print("save graph to %s" % save_path)
    except tf.errors.OutOfRangeError:
        print("reach epoch limit")
    finally:
        coord.request_stop()
    coord.join(threads)
    save_path = saver.save(sess, 'graph.ckpt', global_step=iter)
    
    print("training is done")

with tf.Session() as sess:
    saver.restore(sess, 'graph.ckpt-5000')
    coord_test = tf.train.Coordinator()
    threads_test = tf.train.start_queue_runners(coord=coord_test)
    test_imgs, test_labels = sess.run([test_img_batch, test_label_batch])
    test_imgs = test_imgs.reshape((-1, time_steps, n_input))
    acc = sess.run(accuracy, feed_dict = {x : test_imgs, y_ : test_labels})
    print("predict accuracy is %.2f" % acc)
    coord_test.request_stop()
    coord_test.join(threads_test)

