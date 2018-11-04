import json

import tensorflow as tf

from models.dssm import DSSMEstimator
from preprocess.preprocessor import train_tensor_parser


def train(config):
    def train_input_fn(file_path, seed, batch_size, neg_num, query_len, doc_len, padding):
        """An input function for training
        """
        # Load data set
        data_set = tf.data.TextLineDataset(file_path) \
            .map(lambda line: tf.decode_csv(line, record_defaults=[[0], [''], [''], ['']], field_delim='\t')) \
            .shuffle(buffer_size=1000, seed=seed) \
            .batch(batch_size, drop_remainder=True)

        # next = data_set.make_one_shot_iterator().get_next()
        # sess = tf.Session()
        # sess.run(next)
        # Preprocessor of training set
        data_set = data_set.map(
            lambda _1, _2, _3, _4: train_tensor_parser(batch_size, neg_num, query_len, doc_len, padding, _1, _2, _3, _4)
        )

        # Return the read end of the pipeline.
        batch_query, batch_doc = data_set.make_one_shot_iterator().get_next()
        batch_query.set_shape([None, query_len])
        batch_doc.set_shape([None, doc_len])
        return {'query': batch_query, 'doc': batch_doc}, None

    # Train parameters
    train_param = config['model'].copy()
    train_param.update(config['train'])
    dssm_estimator = DSSMEstimator(model_dir=config['model']['model_folder'], params=train_param)
    dssm_estimator.train(input_fn=lambda: train_input_fn(
        config['train']['file_path'],
        seed=0,
        batch_size=config['train']['batch_size'],
        neg_num=config['model']['neg_num'],
        query_len=config['data']['query_len'],
        doc_len=config['data']['doc_len'],
        padding=config['data']['padding']
    ))


def main():
    with open('./config/dssm.json', 'r') as f_config:
        config = json.load(f_config)
        train(config)


if __name__ == '__main__':
    main()
