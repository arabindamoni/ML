import tensorflow as tf
from word2vec.CbowSimpleHelper import CbowSimpleHelper
import pickle
import time

cbow_helper = CbowSimpleHelper()

learning_rate = 0.01
batch_size = 1000
epochs = 5
display_step = 1
N = 100 # embedding size
V = cbow_helper.get_voc_size()   # vocabulary size

x = tf.placeholder(tf.float32,[None,V])
y = tf.placeholder(tf.float32,[None,V])

def cbow_simple(x, weight):
    hidden_layer = tf.matmul(x, weight['W1'])
    output_layer = tf.matmul(hidden_layer, weight['W2'])
    return output_layer

weight = {
    'W1' : tf.Variable(tf.random_normal(shape=(V,N))),
    'W2' : tf.Variable(tf.random_normal(shape=(N,V)))
}

pred = cbow_simple(x,weight)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    st = time.time()
    avg_cost = 0.0
    total_batch = int(cbow_helper.get_data_size() / batch_size)
    for epoch in range(epochs):
        for i in range(total_batch):
            batch_x,batch_y = cbow_helper.get_batch(i, batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
            # Display logs per epoch step
            print('i: ' + str(i) + ' Time: ' + str(time.time() - st))
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
            "{:.9f}".format(avg_cost))
        print('Finished epoch: '+str(epoch) + 'in Time: '+str(time.time() - st))
    pickle.dump(sess.run(weight['W1']),open('data/weights.pkl'))
print("Optimization Finished!")


