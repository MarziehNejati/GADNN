# SDNE
import tensorflow as tf
import numpy as np
import networkx as nx


tf.compat.v1.disable_eager_execution()

def fc_op(input_op, name, n_out, layer_collector, act_func=tf.nn.leaky_relu):
    n_in = input_op.get_shape()[-1]
    
    
    with tf.compat.v1.name_scope(name) as scope:
        kernel = tf.compat.v1.get_variable(scope + "w", shape=[n_in, n_out],
                                 initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), dtype=tf.float32)
        biases = tf.Variable(tf.constant(0, shape=[1, n_out], dtype=tf.float32), name=scope + 'b')

        fc = tf.add(tf.matmul(input_op, kernel), biases)
        activation = act_func(fc, name=scope + 'act')
        layer_collector.append([kernel, biases])
        return activation


class SDNE(object):
    def __init__(self, graph, encoder_layer_list, alpha=0.01, beta=10., nu=1e-5,
                 batch_size=100, max_iter=500, learning_rate=0.01, adj_mat=None):

        self.g = graph

        self.node_size = self.g.G.number_of_nodes()
        self.rep_size = encoder_layer_list[-1]

        self.encoder_layer_list = [self.node_size]
        self.encoder_layer_list.extend(encoder_layer_list)
        self.encoder_layer_num = len(encoder_layer_list)+1

        self.alpha = alpha
        self.beta = beta
        self.nu = nu
        self.bs = batch_size
        self.max_iter = max_iter
        self.lr = learning_rate

        self.sess = tf.compat.v1.Session()
        self.vectors = {}

        self.adj_mat = nx.to_numpy_array(self.g.G)
        self.embeddings = self.get_train()

        look_back = self.g.look_back_list

        for i, embedding in enumerate(self.embeddings):
            self.vectors[look_back[i]] = embedding

    def get_train(self):
        adj_mat = self.adj_mat

        AdjBatch = tf.compat.v1.placeholder(tf.float32, [None, self.node_size], name='adj_batch')
        Adj = tf.compat.v1.placeholder(tf.float32, [None, None], name='adj_mat')
        B = tf.compat.v1.placeholder(tf.float32, [None, self.node_size], name='b_mat')

        fc = AdjBatch
        scope_name = 'encoder'
        layer_collector = []

        with tf.compat.v1.name_scope(scope_name):
            for i in range(1, self.encoder_layer_num):
                print(("encoder" + str(i)))
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)

        _embeddings = fc

        scope_name = 'decoder'
        with tf.compat.v1.name_scope(scope_name):
            for i in range(self.encoder_layer_num-2, 0, -1):
                print(("decoder" + str(i)))
                fc = fc_op(fc,
                           name=scope_name+str(i),
                           n_out=self.encoder_layer_list[i],
                           layer_collector=layer_collector)
            fc = fc_op(fc,
                       name=scope_name+str(0),
                       n_out=self.encoder_layer_list[0],
                       layer_collector=layer_collector,)

        _embeddings_norm = tf.reduce_sum(tf.square(_embeddings), 1, keepdims=True)

        L_1st = tf.reduce_sum(
            Adj * (
                    _embeddings_norm - 2 * tf.matmul(
                        _embeddings, tf.transpose(_embeddings)
                    ) + tf.transpose(_embeddings_norm)
            )
        )

        L_2nd = tf.reduce_sum(tf.square((AdjBatch - fc) * B))

        L = L_2nd + self.alpha * L_1st

        for param in layer_collector:
            L += self.nu * (tf.reduce_sum(tf.square(param[0]) + tf.abs(param[0])))

        optimizer = tf.compat.v1.train.AdamOptimizer()

        train_op = optimizer.minimize(L)

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        for step in range(self.max_iter):
            index = np.random.randint(self.node_size, size=self.bs)
            adj_batch_train = adj_mat[index, :]
            adj_mat_train = adj_batch_train[:, index]
            b_mat_train = 1.*(adj_batch_train <= 1e-10) + self.beta * (adj_batch_train > 1e-10)

            self.sess.run(train_op, feed_dict={AdjBatch: adj_batch_train,
                                               Adj: adj_mat_train,
                                               B: b_mat_train})
            if step % 20 == 0:
                print(("step %i: %s" % (step, self.sess.run([L, L_1st, L_2nd],
                                                           feed_dict={AdjBatch: adj_batch_train,
                                                                      Adj: adj_mat_train,
                                                                      B: b_mat_train}))))

        return self.sess.run(_embeddings, feed_dict={AdjBatch: adj_mat})

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors)
        fout.write("{} {}\n".format(node_num, self.rep_size))
        for node, vec in list(self.vectors.items()):
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()
