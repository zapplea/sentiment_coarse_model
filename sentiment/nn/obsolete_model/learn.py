def main():
    # sentences_num: the maximum number of sentence in review.
    # words_num: the maximum number of words in a sentence.
    init={'sentences_num':10,'words_num':10,'batch_size':30}
    data_config={'sentences_num':init['sentences_num'],
                 'sentence_words_num':init['sentence_words_num'],
                 'aspect_words_num':init['aspect_words_num'],
                 'batch_size':init['batch_size']}
    rnn_config={}