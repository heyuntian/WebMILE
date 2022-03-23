import numpy as np
# import pymp.config
import scipy.sparse as sp
from scipy.special import softmax
from utils import graph_to_adj, normalized
import tensorflow as tf
from tensorflow.keras import activations, regularizers, constraints, initializers


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse and dense).
    From GitHub @jiongqian/MILE
    https://github.com/jiongqian/MILE/blob/master/refine_model.py
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class scale_weights(tf.keras.constraints.Constraint):

    def __call__(self, w):
        return w - tf.reduce_mean(w, axis=0)


class GCNConv(tf.keras.layers.Layer):
    """
    From Github @cshjin/GCN-TF2.0
    https://github.com/cshjin/GCN-TF2.0/blob/15232a7da73dbca0591a0f8551d7b0fc4495f3de/models/layers.py
    """
    def __init__(self,
                 output_dim,
                 activation=lambda x: x,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 activity_regularizer=None,
                 **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def build(self, input_shape):
        """ GCN has two inputs : [shape(An), shape(X)]
        """
        # gsize = input_shape[0][0]  # graph size
        input_dim = input_shape[1][1]  # feature dim

        if not hasattr(self, 'weight'):
            self.weight = self.add_weight(name="weight",
                                          shape=(input_dim, self.output_dim),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          constraint=self.kernel_constraint,
                                          trainable=True)
        if self.use_bias:
            if not hasattr(self, 'bias'):
                self.bias = self.add_weight(name="bias",
                                            shape=(self.output_dim,),
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint,
                                            trainable=True)
        super(GCNConv, self).build(input_shape)

    def call(self, inputs, **kwargs):
        """ GCN has two inputs : [An, X]
        :param **kwargs:
        """
        self.An = inputs[0]
        self.X = inputs[1]

        output = dot(self.An, dot(self.X, self.weight, sparse=isinstance(self.X, tf.SparseTensor)), sparse=True)

        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)

        if self.activation:
            output = self.activation(output)

        return output


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_to_gcn_adj(adj, lda):  # D^{-0.5} * A * D^{-0.5} : normalized, symmetric convolution operator.
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    self_loop_wgt = np.array(adj.sum(1)).flatten() * lda  # self loop weight as much as sum. This is part is flexible.
    adj_normalized = normalize_adj(adj + sp.diags(self_loop_wgt)).tocoo()
    return adj_normalized


def convert_sparse_matrix_to_sparse_tensor(X):
    if not sp.isspmatrix_coo(X):
        X = X.tocoo()
    indices = np.mat([X.row, X.col]).transpose()
    return tf.SparseTensor(indices, X.data, X.shape)


class GCN(tf.keras.Model):
    """
    Normal GCN with no fairness loss.
    """

    def __init__(self, ctrl):
        super().__init__()
        # Utils and hyperparameters
        self.logger = ctrl.logger
        self.embed_dim = ctrl.embed_dim
        self.act_func = ctrl.refine_model.act_func
        self.wgt_decay = ctrl.refine_model.wgt_decay
        self.regularized = ctrl.refine_model.regularized
        self.learning_rate = ctrl.refine_model.learning_rate
        self.hidden_layer_num = ctrl.refine_model.hidden_layer_num
        self.lda = ctrl.refine_model.lda
        self.epoch = ctrl.refine_model.epoch
        self.early_stopping = ctrl.refine_model.early_stopping
        self.optimizer = ctrl.refine_model.tf_optimizer(learning_rate=self.learning_rate)
        self.ctrl = ctrl

        # Layers
        self.conv_layers = []
        for i in range(self.hidden_layer_num):
            conv = GCNConv(self.embed_dim, activation=self.act_func, use_bias=False,
                           kernel_regularizer=regularizers.l2(l2=self.wgt_decay / 2.0) if self.regularized else None)
            self.conv_layers.append(conv)

    def call(self, gcn_A, input_embed):
        curr = input_embed
        for i in range(self.hidden_layer_num):
            curr = self.conv_layers[i]([gcn_A, curr])
        output = tf.nn.l2_normalize(curr, axis=1)
        return output

    def train(self, coarse_graph=None, fine_graph=None, coarse_embed=None, fine_embed=None):
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))

        if coarse_embed is not None:
            initial_embed = fine_graph.C.dot(coarse_embed)  # projected embedings.
        else:
            initial_embed = fine_embed
        self.logger.info(f'initial_embed: {initial_embed.shape}')
        self.logger.info(f'fine_embed: {fine_embed.shape}')

        loss_arr = []
        for i in range(self.epoch):
            with tf.GradientTape() as tape:
                pred_embed = self.call(struc_A, initial_embed)
                acc_loss = tf.compat.v1.losses.mean_squared_error(fine_embed,
                                                                  pred_embed) * self.embed_dim  # tf.keras.losses.mean_squared_error(y_true=fine_embed, y_pred=pred_embed) * self.embed_dim
                loss = acc_loss
                # print(f'Epoch {i}, Loss: {loss}, Acc Loss: {acc_loss}')
                loss_arr.append(loss)
            grads = tape.gradient(loss, self.variables)
            self.optimizer.apply_gradients(grads_and_vars=zip(grads, self.variables))

    def predict(self, coarse_graph=None, fine_graph=None, coarse_embed=None, last_level=False):
        adj = graph_to_adj(fine_graph)
        struc_A = convert_sparse_matrix_to_sparse_tensor(preprocess_to_gcn_adj(adj, self.lda))
        initial_embed = fine_graph.C.dot(coarse_embed)
        return self.call(struc_A, initial_embed)


def refine(ctrl, graph):
    embedding = normalized(np.loadtxt(ctrl.coarsen_embed), per_feature=False)
    coarse_embed = None
    tf.config.threading.set_intra_op_parallelism_threads(ctrl.workers)
    model = GCN(ctrl)
    model.train(coarse_graph=graph.coarser, fine_graph=graph, coarse_embed=coarse_embed,
                fine_embed=embedding)

    count_lvl = ctrl.coarsen_level
    while graph.finer is not None:  # apply the refinement model.
        embedding = model.predict(coarse_graph=graph, fine_graph=graph.finer, coarse_embed=embedding,
                                  last_level=(count_lvl == 1))
        graph = graph.finer
        ctrl.logger.info("\t\t\tRefinement at level %d completed." % count_lvl)
        count_lvl -= 1
    embedding = embedding.numpy()
    with open(ctrl.embeddings, 'wb') as f:
        np.savetxt(f, embedding)