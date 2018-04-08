import json
import numpy as np
import pickle

class Node:
    def __init__(self):
        self.text = '__empty__'
        self.relation = '__empty__'
        self.index = '__empty__'  # index of the current word in the sentence
        self.parent = '__empty__'
        self.children = []

    def __str__(self):
        return self.text


# TODO: if we use table generated by this class, then we don't need lookup table to get relative distance, since we
# TODO: have already got one.
# TODO: this class produce dependency path and tokenize sentences.
class DependencyGenerator:
    def __init__(self, nn_config, data_config):
        self.nn_config = nn_config
        self.data_config = data_config
        self.dp_result = self.load()
        self.dictionary = self.load_dictionary()

    def load_dictionary(self):
        with open(self.data_config['dictionary_filePath'],'r') as f:
            dic = json.load(f)
        return dic

    def load(self):
        """
        :return:{0:['dp(word-1,word-2)',...],...} 
        """
        with open(self.data_config['dependency_parsing_filePath'],'r') as f:
            dp_result = json.load(f)
        return dp_result

    def recover_original_sentence(self, relations):
        """
        The result will contain 'ROOT' at index=0
        :param relations: [{'rel','parent','child'}, ...]
        :return: 
        """
        sentence = {}
        for relation in relations:
            parent = relation['parent']
            sentence[parent['index']] = parent['word']

            child = relation['child']
            sentence[child['index']] = child['word']
        result = []
        for i in range(len(sentence)):
            result.append(sentence[i])
        return result

    def recover_dp_relation(self, dp_result):
        """

        :param dp_result: dependency relationship between two words, but type is str
        :return: 
        """
        index = dp_result.find('(')
        relation = dp_result[:index]
        # words = 'word1-index1, word2-index2'
        words = dp_result[index + 1:-1]
        # ls = ['word1-index1,','word2-index2']
        ls = words.split()
        # delete ',' in the first item
        ls[0] = ls[0][:-1]
        words = []
        for s in ls:
            index = s.rfind('-')
            words.append({'word': s[:index], 'index': int(s[index + 1:])})

        return {'rel': relation, 'parent': words[0], 'child': words[1]}

    def construct_tree(self, relations, sentence):
        """

        :param relations: 
        :return: 
        """
        tree = [None] * len(sentence)
        for relation in relations:
            rel = relation['rel']
            parent = relation['parent']
            child = relation['child']

            # parent
            if tree[parent['index']] == None:
                node = Node()
                node.text = parent['word']
                node.index = parent['index']

                node.children.append(child['index'])

                tree[node.index] = node
            else:
                node = tree[parent['index']]
                node.children.append(child['index'])

            # child
            if tree[child['index']] == None:
                node = Node()
                node.text = child['word']
                node.index = child['index']

                node.relation = rel
                node.parent = parent['index']

                tree[node.index] = node
            else:
                # previously it only appeared as root
                node = tree[child['index']]
                node.relation = rel
                node.parent = parent['index']
        return tree

    def index_calibrator(self,relations):
        # original index
        org_indexes = set()
        for relation in relations:
            parent = relation['parent']
            child = relation['child']
            org_indexes.add(parent['index'])
            org_indexes.add(child['index'])
        index_list = sorted(list(org_indexes))
        gap=[]
        for i in range(len(index_list)-1):
            if i == 0:
                continue
            cur = index_list[i]
            target = index_list[i+1]
            while (cur+1)<target:
                gap.append(cur)
                cur+=1

        for relation in relations:
            parent = relation['parent']
            child = relation['child']
            for gap_index in gap:
                if parent['index']>gap_index:
                    parent['index']=parent['index']-1
                else:
                    break
            for gap_index in gap:
                if child['index']>gap_index:
                    child['index']=child['index']-1
                else:
                    break
        return relations

    def construct_forest(self):
        forest = []
        sentences = []
        for i in range(len(self.dp_result)):
            instance = self.dp_result[str(i)]
            relations = []
            for relation in instance:
                relations.append(self.recover_dp_relation(relation))
            # TODO: since dependency parser will delete punctuations in the word, so, need to modify the index.
            print('========')
            print(relations)
            print('++++++++')
            relations = self.index_calibrator(relations)
            print(relations)
            print('========')
            # sentence = ['ROOT',word1, word2, ...]
            sentence = self.recover_original_sentence(relations)
            tree = self.construct_tree(relations, sentence)
            sentences.append(sentence)
            forest.append(tree)
        return forest, sentences

    def path_to_root(self, tree, node):
        path = []
        while node.text != 'ROOT':
            parent_index = node.parent
            parent = tree[parent_index]
            path.insert(0, parent)
            node = parent
        return path

    def path_between_nodes(self, path1, path2):
        if len(path1) > len(path2):
            length = len(path1)
        else:
            length = len(path2)
        path = []
        for i in range(length):
            node1 = path1[i]
            node2 = path2[i]
            if node1.index == node2.index:
                # index of last common nodes
                index = i
            else:
                break
        path1 = path1[index:]
        # child <-- relation -->parent<-- ...-->grand_parent
        path1 = self.path_decoder(path1)

        path2 = path2[index:]
        # child <-- relation -->parent<-- ...-->grand_parent
        path2 = self.path_decoder(path2)

        path = []
        path.extend(path1)
        path2.reverse()
        path.extend(path2[1:])
        return path

    def path_decoder(self, path):
        """
        decode node path to words$relation path
        :param path: from parent node to child node
        :return: 
        """
        # from child node to parent node
        path = path.reverse()
        ls = []
        # False: word to stack then relation to stack
        # True: relation to stack then word to stack
        for i in range(len(path)):
            node = path[i]
            if node.text == "ROOT" or i == len(path)-1:
                ls.append(node.text)
            else:
                ls.append(node.text)
                ls.append(node.relation)
        return ls

    def relative_distance_table_generator(self, tree):
        """
        Construct lookup table for each relation, the index is (start word index, target word index). The index is the position of the
        word in the sentence.
        :param relations: tree: [Node,...]
        :return: 
        """
        # the key is sentiment word
        tables = {}
        max_path_length = 0
        max_table_length = 0
        for i in range(len(tree)):
            if i == 0:
                continue
            node = tree[i]
            sentiment_node = node[i]
            sentiment_node_index = i
            sentiment_word_index = i - 1
            sentiment_node_path = self.path_to_root(tree, sentiment_node)
            # key is relative distance
            table = []
            for j in range(len(tree)):
                if j == 0:
                    continue
                attribute_node = node[j]
                attribute_node_index = j
                attribute_word_index = j - 1
                attribute_node_path = self.path_to_root(tree, attribute_node)
                path = self.path_between_nodes(sentiment_node_path, attribute_node_path)
                if max_path_length < len(path):
                    max_path_length = len(path)
                table.append(path)
            if len(table)>max_table_length:
                max_table_length=len(table)
            tables[sentiment_word_index] = table
        return tables, max_path_length, max_table_length

    def tables_encoder(self, tables, max_path_length,max_table_length,max_sentence_length):
        """
        convert words in table to word
        :param tables: dict 
        :return: 
        """
        encoded_tables = []
        for j in range(len(tables)):
            encoded_table = []
            table = tables[j]
            for i in range(len(table)):
                path = table[i]
                encoded_path = []
                for word in path:
                    if word in self.dictionary:
                        encoded_path.append(self.dictionary[word])
                    else:
                        encoded_path.append(self.dictionary['#UNK#'])
                while len(encoded_path) < max_path_length:
                    encoded_path.append(self.nn_config['padding_word_index'])
                encoded_table.append(encoded_path)
            while len(encoded_table)<max_table_length:
                encoded_table.append([self.nn_config['padding_word_index']]*max_path_length)
            encoded_tables.append(np.array(encoded_table,dtype='int32'))
        while len(encoded_tables)<max_sentence_length:
            encoded_tables.append(np.ones(shape=(max_table_length,max_path_length),dtype='int32')*self.nn_config['padding_word_index'])
        return encoded_tables

    def sentences_encoder(self, sentences):
        max_sentence_length = 0
        for sentence in sentences:
            if len(sentence) > max_sentence_length:
                max_sentence_length = len(sentence)
        max_sentence_length = max_sentence_length - 1
        encoded_sentences = []
        for sentence in sentences:
            # TODO: delete 'ROOT' in sentence
            sentence = sentence[1:]
            encoded_sentence = []
            for word in sentence:
                if word in self.dictionary:
                    encoded_sentence.append(self.dictionary[word])
                else:
                    encoded_sentence.append(self.dictionary['#UNK#'])
            while len(encoded_sentence) < max_sentence_length:
                encoded_sentence.append(self.nn_config['padding_word_index'])
            encoded_sentences.append(encoded_sentence)
        return encoded_sentences,max_sentence_length


    def write(self,encoded_tables,encoded_sentences):
        with open(self.data_config['relative_distance_table'],'wb') as f:
            pickle.dump({'encoded_tables':encoded_tables,'encoded_sentences':encoded_sentences},f)


    def main(self):
        # sentences contain ROOT
        forest, sentences = self.construct_forest()
        # max_sentence_length decide how many tables should a sentence contain
        encoded_sentences,max_sentence_length = self.sentences_encoder(sentences)
        tables = []
        max_path_length = 0
        max_table_length =0
        for tree in forest:
            table, path_length,table_length = self.relative_distance_table_generator(tree)
            tables.append(table)
            if path_length > max_path_length:
                max_path_length = path_length
            if table_length > max_table_length:
                max_table_length = table_length
        encoded_tables = []
        print('max_path_length: ',max_path_length)
        for table in tables:
            encoded_tables.append(self.tables_encoder(table, max_path_length,max_table_length,max_sentence_length))
        self.write(encoded_tables,encoded_sentences)

if __name__ == '__main__':
    data_configs = [{
                        'dependency_parsing_filePath':'/datastore/liu121/senti_data/pd/path_dependency_resturant_train.json',
                        'relative_distance_table':'/datastore/liu121/senti_data/pd/train_pd.table',
                        'dictionary_filePath':'/datastore/liu121/senti_data/en_word2id_dictionary.json'},
                   {
                       'dependency_parsing_filePath': '/datastore/liu121/senti_data/pd/path_dependency_resturant_test.json',
                       'relative_distance_table':'/datastore/liu121/senti_data/pd/test_pd.table',
                       'dictionary_filePath': '/datastore/liu121/senti_data/en_word2id_dictionary.json'}
                   ]

    nn_config = {'padding_word_index': 0}
    for data_config in data_configs:
        pd_gen = DependencyGenerator(nn_config,data_config)
        pd_gen.main()
    print('finish')