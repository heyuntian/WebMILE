import numpy as np
import tensorflow as tf

class MILEAPIControl:
    def __init__(self):
        """
        Task information
        """
        self.root = 'jobs'
        self.jobid = 'test'
        self.input_format = 'edgelist'
        self.output_format = 'edgelist'
        self.path = 'jobs/test/'
        self.graph_path = f'{self.path}/graph.edgelist'
        self.coarsen_path = f'{self.path}/coarsened_graph.edgelist'
        self.coarsen_embed = f'{self.path}/coarsened_embeddings.npy'
        self.embeddings = f'{self.path}/embeddings.npy'
        """
        Graph Coarsening parameters
        """
        self.coarsen_to = 500  # 1000
        self.coarsen_level = 0  #
        self.max_node_wgt = 100  # to avoid super-node being too large.
        """
        Embedding and Refinement parameters
        """
        self.embed_dim = 128
        self.language = 'python'
        self.refine_model = RefineModelSetting()

    def resetTaskInfo(self, useEmbed=False):
        while self.root and self.root[-1] == '/':
            self.root = self.root[:-1]
        while self.jobid and self.jobid[-1] == '/':
            self.jobid = self.jobid[:-1]
        self.path = f'{self.root}/{self.jobid}'
        self.graph_path = f'{self.path}/graph.{self.input_format}'
        self.coarsen_path = f'{self.path}/coarsened_graph.{self.output_format}'
        self.coarsen_embed = f'{self.path}/coarsened_embeddings.txt'
        self.embeddings = f'{self.path}/embeddings.txt'


class RefineModelSetting:
    def __init__(self):
        self.double_base = False
        self.learning_rate = 0.001
        self.epoch = 200
        self.early_stopping = 50  # Tolerance for early stopping (# of epochs).
        self.wgt_decay = 5e-4
        self.regularized = True
        self.hidden_layer_num = 2
        self.act_func = tf.keras.activations.tanh   # tf.tanh
        self.tf_optimizer = tf.keras.optimizers.Adam
        self.lda = 0.05  # self-loop weight lambda
        self.lambda_fl = 1
        self.negative = 50
        self.fair_threshold = 0.5


# Deprecated: Implementation in TensorFlow v1
# # A Control instance stores most of the configuration information.
# import tensorflow as tf
# class Control:
#     def __init__(self):
#         self.data = None
#         self.workers = 4
#         self.coarsen_to = 500  # 1000
#         self.coarsen_level = 0  #
#         self.max_node_wgt = 100  # to avoid super-node being too large.
#         self.embed_dim = 128
#         self.basic_embed = "DEEPWALK"
#         self.refine_type = "MD-gcn"
#         self.refine_model = RefineModelSetting()
#         self.embed_time = 0.0  # keep track of the amount of time spent for embedding.
#         self.debug_mode = False  # set to false for time measurement.
#         self.logger = None
#
#
# class RefineModelSetting:
#     def __init__(self):
#         self.double_base = False
#         self.learning_rate = 0.001
#         self.epoch = 200
#         self.early_stopping = 50  # Tolerance for early stopping (# of epochs).
#         self.wgt_decay = 5e-4
#         self.regularized = True
#         self.hidden_layer_num = 2
#         self.act_func = tf.tanh
#         self.tf_optimizer = tf.train.AdamOptimizer
#         self.lda = 0.05  # self-loop weight lambda
#         self.untrained_model = False  # if set. The model will be untrained.
#
#         # The following ones are for GraphSAGE only.
#         self.gs_sample_neighbrs_num = 100
#         self.gs_mlp_layer = 2
#         self.gs_concat = True
#         self.gs_uniform_sample = False
#         self.gs_self_wt = True
