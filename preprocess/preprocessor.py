import tensorflow as tf


def train_tensor_parser(
        batch_size, neg_num, query_len, doc_len, padding,
        id_tensor, query_tensor, pos_doc_tensor, neg_doc_tensor
):
    """
    Parser of training set

    :param batch_size: Batch size.
    :param neg_num: Number of negative sample.
    :param query_len: Limited query length.
    :param doc_len: Limited doc length.
    :param padding: Pad to the end.

    :param id_tensor: Tensor of line id.
    :param query_tensor: Tensor of query.
    :param pos_doc_tensor: Tensor of positive sample.
    :param neg_doc_tensor: Tensor of negative sample.
    :return:
    """

    def __parser(id_list, query_str_list, pos_doc_str_list, neg_doc_str_list):
        batch_query = []
        batch_doc = []

        for i in range(batch_size):
            # Query
            query_str = query_str_list[i]
            query = list(map(int, query_str.decode('utf-8').split(',')))
            batch_query.extend([query] * (neg_num + 1))

            # Positive doc
            pos_doc_str = pos_doc_str_list[i]
            pos_doc = list(map(int, pos_doc_str.decode('utf-8').split(',')))
            batch_doc.append(pos_doc)

            # Negative doc
            neg_doc_str = neg_doc_str_list[i]
            neg_docs = neg_doc_str.decode('utf-8').split(';')
            assert len(neg_docs) == neg_num, 'Need {} negative samples, but provides {} in id {}'.format(
                str(neg_num), str(len(neg_docs)), str(id_list[i])
            )
            for neg_doc in neg_docs:
                neg_doc = list(map(int, neg_doc.split(',')))
                batch_doc.append(neg_doc)

        batch_query = pad_or_truncate(batch_query, query_len, padding)
        batch_doc = pad_or_truncate(batch_doc, doc_len, padding)
        return batch_query, batch_doc

    return tf.py_func(
        __parser, inp=[id_tensor, query_tensor, pos_doc_tensor, neg_doc_tensor], Tout=(tf.int64, tf.int64)
    )


def pad_or_truncate(sequences, limited_len, padding):
    """
    Padding or truncate list to limited length

    :param sequences: List to be processed
    :param limited_len: Limited length
    :param padding: Value to be padded at the end.
    """
    result = []
    for sequence in sequences:
        result.append(
            sequence[0:limited_len]
            if len(sequence) > limited_len else sequence + [padding] * (limited_len - len(sequence))
        )
    return result
