"""
Generate fake toy data.
Purpose of these codes are just showing the data format.
"""
import random


def fake_train(file_path, vocab_size, query_max_len, doc_max_len, sample_num, neg_num):
    """
    Generate fake train date, each word of sample is concatenated by comma. The format likes:
    line_id query pos_sample neg_sample;neg_sample;...

    :param file_path: Train file path.
    :param vocab_size: Vocabulary size.
    :param query_max_len: Maximum of query length.
    :param doc_max_len: Maximum of doc length.
    :param sample_num: Number of total samples.
    :param neg_num: Number of negative samples.
    """
    with open(file_path, 'w+') as f_out:
        for i in range(sample_num):
            # Generate fake query
            query_len = random.randint(query_max_len // 3, query_max_len)
            query = ','.join(
                map(str, random.sample(range(1, vocab_size), query_len))
            )

            # Generate fake docs
            doc_len = random.randint(doc_max_len // 2, doc_max_len)
            pos_doc = ','.join(
                map(str, random.sample(range(1, vocab_size), doc_len))
            )
            neg_docs = []
            for neg in range(neg_num):
                neg_doc = random.sample(range(1, vocab_size), doc_len)
                neg_docs.append(','.join(map(str, neg_doc)))

            f_out.write(
                '{}\t{}\t{}\t{}\n'.format(str(i), query, pos_doc, ';'.join(neg_docs))
            )


def fake_query_test(
        file_path, vocab_size, query_max_len, doc_max_len, sample_num, max_doc_num_per_query, most_relevance_score
):
    """
    Generate fake query-doc search relevance date.
    Each word of sample is concatenated by comma. The format likes:
    line_id query doc_id-doc-score;doc_id-doc-score;...

    :param file_path: Train file path.
    :param vocab_size: Vocabulary size.
    :param query_max_len: Maximum of query length.
    :param doc_max_len: Maximum of doc length.
    :param sample_num: Number of test query.
    :param max_doc_num_per_query: Maximum number of docs per query.
    :param most_relevance_score: Maximum relevance score between query and doc.
    """
    with open(file_path, 'w+') as f_out:
        for i in range(sample_num):
            # Generate fake query
            query_len = random.randint(query_max_len // 3, query_max_len)
            query = ','.join(
                map(str, random.sample(range(1, vocab_size), query_len))
            )

            # Generate fake doc
            doc_num = random.randint(max_doc_num_per_query // 3, max_doc_num_per_query)
            docs = []
            for num in range(doc_num):
                doc_len = random.randint(doc_max_len // 3, doc_max_len)
                doc_str = ','.join(
                    map(str, random.sample(range(1, vocab_size), doc_len))
                )
                doc_score = random.randint(0, most_relevance_score)
                doc = '{}-{}-{}'.format(str(num), doc_str, str(doc_score))
                docs.append(doc)

            # Write to file
            f_out.write(
                '{}\t{}\t{}\n'.format(str(i), query, ';'.join(docs))
            )


if __name__ == '__main__':
    fake_train(
        'train.csv', vocab_size=4000, query_max_len=10, doc_max_len=40, sample_num=20, neg_num=4
    )
    fake_query_test(
        'query_test.csv', vocab_size=4000, query_max_len=10, doc_max_len=40, sample_num=20,
        max_doc_num_per_query=20, most_relevance_score=3
    )
