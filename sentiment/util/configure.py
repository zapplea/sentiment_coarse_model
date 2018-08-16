
config = {
        # 'coarse_train_source_file': '/hdd/lujunyu/dataset/sentiment_coarse_model/coarse/yelp_lda_trainset.pkl',
        # 'coarse_test_source_file': '/hdd/lujunyu/dataset/sentiment_coarse_model/coarse/yelp_lda_testset.pkl',
        'coarse_train_source_file': '/hdd/lujunyu/acl2018/yelp/yelp_lda_trainset.pkl',
        'coarse_test_source_file': '/hdd/lujunyu/acl2018/yelp/yelp_lda_testset.pkl',
        'coarse_train_data_file': '/hdd/lujunyu/dataset/sentiment_coarse_model/coarse/train_data.pkl',
        'coarse_test_data_file': '/hdd/lujunyu/dataset/sentiment_coarse_model/coarse/test_data.pkl',
        'fine_train_source_file': '/hdd/lujunyu/dataset/sentiment_coarse_model/fine/absa_resturant_train.pkl',
        'fine_test_source_file': '/hdd/lujunyu/dataset/sentiment_coarse_model/fine/absa_resturant_test.pkl',
        'fine_train_data_file': '/datastore/liu121/sentidata2/expdata/fine_data/fine_train_data.pkl', # '/hdd/lujunyu/dataset/sentiment_coarse_model/fine/train_data.pkl',
        'fine_test_data_file': '/datastore/liu121/sentidata2/expdata/fine_data/fine_test_data.pkl',#'/hdd/lujunyu/dataset/sentiment_coarse_model/fine/test_data.pkl',
        'wordembedding_file_path': '/hdd/lujunyu/dataset/glove/glove.42B.300d.pkl',
        #'stopwords_file_path': '~/dataset/semeval2016/stopwords.txt',
        #'dictionary': '/hdd/lujunyu/dataset/sentiment_coarse_model/data_dictionary_old.pkl',



        'lstm_cell_size': 300,
        'word_dim': 300,
        'attribute_dim': 300,
        'lookup_table_words_num': 34934,  # 34934,2074276 for Chinese word embedding
        'padding_word_index': 34933,  # 34933,the index of #PAD# in word embeddings list
        'epoch': 1000,
        'words_num': 40,
        'atr_pred_threshold': 0,
        'is_mat': True,

}