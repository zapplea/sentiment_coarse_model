import numpy as np


def print_prediction(pred_vec, aspect, index):
    pred_check = []
    for r in index:
        print(r,len(pred_vec))
        for c, rr in enumerate(pred_vec[r]):
            if rr:
                pred_check.append(aspect[c])
    print('Prediction: ', pred_check)

def print_label(dg, true_label, index):
    label = []
    for r in index:
        for c, rr in enumerate(true_label[r]):
            if rr:
                label.append(list(dg.aspect_dic.keys())[c])
    print('Label: ', label)

def print_sentence(dg, vocab, index, flag):
    sentence = []
    for r in index:
        if flag == 'train':
            for sent in dg.train_sentence[r]:
                for word in sent:
                    if word != 34933:
                        sentence.append(vocab[word])
        elif flag == 'test':
            for sent in dg.test_sentence[r]:
                for word in sent:
                    if word != 34933:
                        sentence.append(vocab[word])
    print('Review: ', ' '.join(sentence))

def print_aspect_score(score, aspect, index):
    for r in index:
        for i in range(6):
            print(aspect[i],':',score[r][i],' ')

def print_word_score(dg, score, aspect, vocab, index, flag):
    for r in index:
        for m, a_s in enumerate(score[r]):
            for i, r_s in enumerate(a_s):
                for j, w_s in enumerate(r_s):
                    if w_s > 0 and flag == 'train':
                        print(aspect[i],': ',vocab[dg.train_sentence[r][m][j]],'[%f]' % w_s)
                    elif w_s > 0 and flag == 'test':
                        print(aspect[i],': ',vocab[dg.test_sentence[r][m][j]],'[%f]' % w_s)

def visualization_train(dg,vocab ,aspect_list,true_label,pred_vec,score_vec,score_pre_vec ,epoch,mt,
                        loss_vec, TP_vec, FP_vec, FN_vec):

    # np.random.seed(1)
    check_num = 1


    print('Epoch:', epoch, '\nTraining loss:%.10f' % np.mean(loss_vec))

    _precision = mt.precision(TP_vec,FP_vec,'macro')
    _recall = mt.recall(TP_vec,FN_vec,'macro')
    _f1_score = mt.f1_score(_precision,_recall,'macro')
    print('F1 score for each class:',_f1_score,'\nPrecision for each class:',_precision,'\nRecall for each class:',_recall)
    print('Macro F1 score:',np.mean(_f1_score) ,' Macro precision:', np.mean(_precision),' Macro recall:', np.mean(_recall) )

    _precision = mt.precision(TP_vec, FP_vec, 'micro')
    _recall = mt.recall(TP_vec, FN_vec, 'micro')
    _f1_score = mt.f1_score(_precision, _recall, 'micro')
    print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision), ' Micro recall:', np.mean(_recall))

    # random_display = np.random.randint(0, 1300, check_num)
    # print_sentence(dg, vocab, random_display,'train')
    # print_prediction(pred_vec, aspect_list,random_display)
    # print_label(dg, true_label, random_display)
    # print_aspect_score(score_vec, aspect_list,random_display)
    # print_word_score(dg, score_pre_vec, aspect_list, vocab, random_display,'train')

def visualization_test(dg, vocab, true_label, aspect_list, epoch, mt, loss_vec, TP_vec, FP_vec, FN_vec):

    # np.random.seed(1)
    check_num = 1

    print('\nTest loss:%.10f' % np.mean(loss_vec))

    _precision = mt.precision(TP_vec, FP_vec, 'macro')
    _recall = mt.recall(TP_vec, FN_vec, 'macro')
    _f1_score = mt.f1_score(_precision, _recall, 'macro')
    # print('F1 score for each class:', _f1_score, '\nPrecision for each class:', _precision,
    #       '\nRecall for each class:', _recall)
    print('Macro F1 score:', np.mean(_f1_score), ' Macro precision:', np.mean(_precision),
          ' Macro recall:', np.mean(_recall))

    _precision = mt.precision(TP_vec, FP_vec, 'micro')
    _recall = mt.recall(TP_vec, FN_vec, 'micro')
    _f1_score = mt.f1_score(_precision, _recall, 'micro')
    print('Micro F1 score:', _f1_score, ' Micro precision:', np.mean(_precision),
          ' Micro recall:', np.mean(_recall))

    random_display = np.random.randint(0, 400, check_num)
    print_sentence(dg, vocab, random_display,'test')
    # print_prediction(dg, pred_vec, random_display)
    print_label(dg, true_label, random_display)
    # print_aspect_score(score_vec, aspect_list,random_display)
    # print_word_score(dg, score_pre_vec, aspect_list, vocab, random_display,'test')
