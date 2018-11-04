import tensorflow as tf


def likelihood_click_loss(cos_sim, batch_size, neg_num, gamma):
    """
    Loss function for DSSM model.
    The positive doc must be first of each query result.

    :param cos_sim: Cosine similarity between query and doc.
    :param batch_size: Batch size.
    :param neg_num: Number of negative samples.
    :param gamma: Smoothing factor in the soft-max function.
    :return: loss operation
    """
    # Reshape cosine similarity
    cos_sim = tf.reshape(cos_sim, [batch_size, neg_num + 1]) * gamma

    # Compute the posterior probability
    posterior_prob = tf.nn.softmax(cos_sim, axis=1)
    # Soft-max of position document
    pos_prob = tf.slice(posterior_prob, [0, 0], [-1, 1])

    # Likelihood of the clicked documents
    loss = -tf.reduce_sum(tf.log(pos_prob))

    return loss
