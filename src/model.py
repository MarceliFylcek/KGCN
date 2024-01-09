import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score
from typing import List, Dict
from tensorflow.keras.layers import LSTM
import numpy as np
import sys
from LSTM import CustomLSTM

class KGCN(object):
    def __init__(self, args, n_user: int, n_entity: int, n_relation: int, adj_entity: List[List[int]], adj_relation: List[List[int]]):
        """
        #! Neighbours are chosen once and don't change.
        
        Args:
            args (_type_): Input arguments
            n_user (int): number of users
            n_entity (int): number of entities
            n_relation (int): numer of relations
            adj_entity (List[List[int]]): shape = [n_entity, neighbor_sample_size]
            adj_relation (List[List[int]]): shape = [n_relation, neighbor_sample_size]
            time_stamps (int): How far into the past the model sees
        """
        self._parse_args(args, adj_entity, adj_relation)

        self.lstm = CustomLSTM(input_size=self.dim, hidden_size=self.dim)

        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation

        self.time_stamps = args.time_stamps
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        #. Received in every pass through the feed_dict

        #. User indices of length equal to batch size
        self.user_indices = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None], name='user_indices')

        #. Few last items each user has interacted with, shape = [batch_size, time_stamps, 1]
        #. With the last item being the one the prediction is for
        self.item_history = tf.compat.v1.placeholder(dtype=tf.int64, shape=[None, self.time_stamps], name='item_history')

        # Label for the last item in item_history
        self.labels = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None], name='labels')

    def _build_model(self, n_user: int, n_entity: int, n_relation: int):

        #. Embedding for every single user of size self.dim
        self.user_emb_matrix = tf.compat.v1.get_variable(
            shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
        #. Embedding for every single entity of size self.dim
        self.entity_emb_matrix = tf.compat.v1.get_variable(
            shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
        #. Embedding for every single relation of size self.dim
        self.relation_emb_matrix = tf.compat.v1.get_variable(
            shape=[n_relation, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')

        #. Embeddings for every user in the batch [batch_size, dim]
        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        #! Forward pass through the network

        history_embeddings = []

        # [time_stamp, batch]
        history = tf.transpose(self.item_history)

        for i in range(self.time_stamps):

            batch = history[i]

            # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
            # dimensions of entities:
            # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
            entities, relations = self.get_neighbors(batch)

            # [batch_size, dim]
            self.item_embeddings, self.aggregators = self.aggregate(entities, relations)

            history_embeddings.append(self.item_embeddings)

        # [time_steps, batch_size, dim]
        stacked_embeddings = tf.stack(history_embeddings, axis=0)

        # [batch_size, time_steps, dim]
        stacked_embeddings = tf.transpose(stacked_embeddings, perm=[1, 0, 2])
        
        lstm_output = self.lstm(stacked_embeddings)

        self.scores = tf.reduce_sum(self.user_embeddings * lstm_output, axis=1)
        print(self.scores.shape)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

            res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):

        #. Cross-entropy between labels and scores
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        #. L2-loss for the embeddings
        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        
        lstm_weights = self.lstm.trainable_weights
        lstm_l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in lstm_weights])

        # L2-loss for the aggregator weights
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        #. Final loss function
        self.loss = self.base_loss + self.l2_weight * self.l2_loss + lstm_l2_loss

        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict: Dict[List[List[int]], List[List[int]]]):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict: Dict[List[List[int]], List[List[int]]]):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict: Dict[List[List[int]], List[List[int]]]):
        return sess.run([self.item_history, self.scores_normalized], feed_dict)
