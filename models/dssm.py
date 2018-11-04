import tensorflow as tf
import tensorflow.contrib as contrib

from losses.losses_impl import likelihood_click_loss


class DSSM:
    def __init__(self, hidden_units, activation_fn, w_initializer, b_initializer):
        """
        :param hidden_units: List of integer, unit of hidden layer.
        :param activation_fn: Activation function.
        :param w_initializer: Initializer function for the weight matrix.
        :param b_initializer: Initializer function for the bias.
        """
        self.hidden_units = hidden_units
        self.activation_fn = activation_fn
        self.w_initializer = w_initializer
        self.b_initializer = b_initializer

    def __mlp(self, input_tensor):
        # Multi-Layer Perceptron
        for units in self.hidden_units:
            input_tensor = tf.layers.dense(
                input_tensor, units=units, activation=self.activation_fn, use_bias=True,
                kernel_initializer=self.w_initializer, bias_initializer=self.b_initializer
            )
        return input_tensor

    def build(self, query, doc):
        with tf.name_scope('Query'):
            query_emb = self.__mlp(query)
        with tf.name_scope('Doc'):
            doc_emb = self.__mlp(doc)

        # Cosine similarity
        with tf.name_scope('Similarity'):
            query_l2_norm = tf.nn.l2_normalize(query_emb, axis=1)
            doc_l2_norm = tf.nn.l2_normalize(doc_emb, axis=1)

            score = tf.reduce_sum(tf.multiply(query_l2_norm, doc_l2_norm), axis=1)

        return score


class DSSMEstimator(tf.estimator.Estimator):
    def __init__(self, model_dir, params, config=None):
        def __lr_decay_fn(learning_rate, global_step):
            return tf.train.exponential_decay(
                learning_rate=learning_rate, global_step=global_step,
                decay_steps=params['decay_steps'], decay_rate=params['decay_rate'], staircase=True
            )

        def __model_fn(features, labels, mode):
            dssm = DSSM(
                params.get('hidden_units', [300, 128]),
                params.get('activation_fn', tf.nn.tanh),
                params.get('w_initializer', tf.contrib.layers.xavier_initializer()),
                params.get('b_initializer', tf.contrib.layers.xavier_initializer())
            )
            # Get score
            query = tf.cast(features['query'], dtype=tf.float32)
            doc = tf.cast(features['doc'], dtype=tf.float32)
            cos_sim = dssm.build(query, doc)

            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode, predictions=cos_sim)

            # Create train op.
            assert mode == tf.estimator.ModeKeys.TRAIN
            # Add loss function
            loss = likelihood_click_loss(cos_sim, params['batch_size'], params['neg_num'], params['gamma'])

            # Train op
            global_step = tf.Variable(0, trainable=False)
            train_op = contrib.layers.optimize_loss(
                loss=loss,
                global_step=global_step,
                learning_rate=params['learning_rate'],
                optimizer=params['optimizer'],
                learning_rate_decay_fn=__lr_decay_fn
            )
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        super(DSSMEstimator, self).__init__(model_fn=__model_fn, model_dir=model_dir, config=config)
