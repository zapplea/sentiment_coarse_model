if __name__ == "__main__":
    seed = {'lstm_cell_size': 300,
            'word_dim': 300
            }
    nn_config = { # fixed parameter
                   'attributes_num': 12,
                   'sentiment_dim': seed['lstm_cell_size'],  # dim of a sentiment expression prototype.
                   'attribute_dim': seed['lstm_cell_size'],
                   'words_num': 20,
                   'word_dim': seed['word_dim'],
                   'is_mat': True,
                   'epoch': 10000,#10000
                   'batch_size':30,
                   'lstm_cell_size': seed['lstm_cell_size'],
                   'lookup_table_words_num': 3000000,  # 2074276 for Chinese word embedding
                   'padding_word_index': 0,
                   'max_path_length':None,
                   # flexible parameter
                   'attribute_senti_prototype_num': 4,
                   'normal_senti_prototype_num': 4,  # number of specific sentiment of each attribute
                   'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
                   'sentiment_loss_theta': 1.0,
                   'rps_num': 5,  # number of relative positions
                   'rp_dim': 100,  # dimension of relative position
                   'lr': 0.003,  # learing rate
                   'reg_rate': 0.3,
                   'senti_pred_threshold': 0.5,
                   'report_filePath': '/datastore/liu121/nosqldb2/sentiA/report1.txt'
                 }
