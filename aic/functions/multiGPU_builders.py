import tensorflow as tf

class SentiNetBuilder:
    def __init__(self,config):
        self.nn_config = {
            'words_num': 210,
            'lstm_cell_size': 300,
            'clip_value': 3.0,
            'word_dim': 300,
            'attribute_dim': 300,
            'lookup_table_words_num': 116141,  # 34934,2074276 for Chinese word embedding
            'padding_word_index': 116140,  # 34933,the index of #PAD# in word embeddings list
            'attribute_mat_size': 3,  # number of attribute mention prototypes in a attribute matrix
            'attributes_num': 20,  # fine attributes number
            'coarse_attributes_num': 20,
            'atr_pred_threshold': 0,
            'review_atr_pred_threshold': 0.5,
            'max_review_len': 19,
            'normal_senti_prototype_num': 4,
            'attribute_senti_prototype_num': 4,
            'sentiment_dim': 300,
            'sentiment_num':3,
            'rps_num': 5,
            'rps_dim': 100,
            'reg_rate': 1E-5,
            'lr': 1E-4,
            'is_mat': True,
            'gpu_num':2,
            'with_elmo':False,
            'elmo':{}
        }
        self.nn_config.update(config)
        self.nn_config['elmo']['padding_word_index']=self.nn_config['padding_word_index']

    def average_gradients(self, tower_grads):
        # calculate average gradient for each shared variable across all GPUs
        # shape of tower_grads: [((grad0_gpu0, var0_gpu0), (grad1_gpu0,var1_gpu0),...),
        #                        ((grad0_gpu1, var0_gpu1), (grad1_gpu1,var1_gpu1),...),
        #                        ((grad0_gpu2, var0_gpu2), (grad1_gpu2,var1_gpu2),...),
        #                        ...]
        # zip(tower_grads)-->[((grad0_gpu0, var0_gpu0),(grad0_gpu1, var0_gpu1),(grad0_gpu2, var0_gpu2)),
        #                     ((grad1_gpu0, var1_gpu0),(grad1_gpu1, var1_gpu1),(grad1_gpu2, var0_gpu2)),
        #                     ... ...]
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            # We need to average the gradients across each GPU.

            g0, v0 = grad_and_vars[0]

            if g0 is None:
                # no gradient for this variable, skip it
                average_grads.append((g0, v0))
                continue

            if isinstance(g0, tf.IndexedSlices):
                # If the gradient is type IndexedSlices then this is a sparse
                #   gradient with attributes indices and values.
                # To average, need to concat them individually then create
                #   a new IndexedSlices object.
                indices = []
                values = []
                for g, v in grad_and_vars:
                    indices.append(g.indices)
                    values.append(g.values)
                all_indices = tf.concat(indices, 0)
                avg_values = tf.concat(values, 0) / len(grad_and_vars)
                # deduplicate across indices
                av, ai = self._deduplicate_indexed_slices(avg_values, all_indices)
                grad = tf.IndexedSlices(av, ai, dense_shape=g0.dense_shape)

            else:
                # a normal tensor can just do a simple average
                grads = []
                for g, v in grad_and_vars:
                    # Add 0 dimension to the gradients to represent the tower.
                    expanded_g = tf.expand_dims(g, 0)
                    # Append on a 'tower' dimension which we will average over
                    grads.append(expanded_g)

                # Average over the 'tower' dimension.
                grad = tf.concat(grads, 0)
                grad = tf.reduce_mean(grad, 0)

            # the Variables are redundant because they are shared
            # across towers. So.. just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)

            average_grads.append(grad_and_var)

        assert len(average_grads) == len(list(zip(*tower_grads)))

        return average_grads

    def _deduplicate_indexed_slices(self,values, indices):
        """Sums `values` associated with any non-unique `indices`.
        Args:
          values: A `Tensor` with rank >= 1.
          indices: A one-dimensional integer `Tensor`, indexing into the first
          dimension of `values` (as in an IndexedSlices object).
        Returns:
          A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
          de-duplicated version of `indices` and `summed_values` contains the sum of
          `values` slices associated with each unique index.
        """
        unique_indices, new_index_positions = tf.unique(indices)
        summed_values = tf.unsorted_segment_sum(
            values, new_index_positions,
            tf.shape(unique_indices)[0])
        return (summed_values, unique_indices)

    def clip_by_global_norm_summary(self, t_list, clip_norm, norm_name, variables):
        # wrapper around tf.clip_by_global_norm that also does summary ops of norms

        # compute norms
        # use global_norm with one element to handle IndexedSlices vs dense
        norms = [tf.global_norm([t]) for t in t_list]

        # summary ops before clipping
        summary_ops = []
        for ns, v in zip(norms, variables):
            name = 'norm_pre_clip/' + v.name.replace(":", "_")
            summary_ops.append(tf.summary.scalar(name, ns))

        # clip
        clipped_t_list, tf_norm = tf.clip_by_global_norm(t_list, clip_norm)

        # summary ops after clipping
        norms_post = [tf.global_norm([t]) for t in clipped_t_list]
        for ns, v in zip(norms_post, variables):
            name = 'norm_post_clip/' + v.name.replace(":", "_")
            summary_ops.append(tf.summary.scalar(name, ns))

        summary_ops.append(tf.summary.scalar(norm_name, tf_norm))

        return clipped_t_list, tf_norm, summary_ops

    def clip_grads(self, grads, options, do_summaries, global_step):
        # grads = [(grad1, var1), (grad2, var2), ...]
        def _clip_norms(grad_and_vars, val, name):
            # grad_and_vars is a list of (g, v) pairs
            grad_tensors = [g for g, v in grad_and_vars]
            vv = [v for g, v in grad_and_vars]
            scaled_val = val
            if do_summaries:
                clipped_tensors, g_norm, so = self.clip_by_global_norm_summary(
                    grad_tensors, scaled_val, name, vv)
            else:
                so = []
                clipped_tensors, g_norm = tf.clip_by_global_norm(
                    grad_tensors, scaled_val)

            ret = []
            for t, (g, v) in zip(clipped_tensors, grad_and_vars):
                ret.append((t, v))

            return ret, so

        all_clip_norm_val = options['all_clip_norm_val']
        ret, summary_ops = _clip_norms(grads, all_clip_norm_val, 'norm_grad')

        assert len(ret) == len(grads)

        return ret, summary_ops

    def concat_pred_labels(self, graph, gpu_num):
        attr_total_pred_labels = []
        senti_total_pred_labels = []
        joint_total_pred_labels = []
        for i in range(gpu_num):
            attr_total_pred_labels.append(graph.get_collection('attr_pred_labels')[i])
            senti_total_pred_labels.append(graph.get_collection('senti_pred_labels')[i])
            joint_total_pred_labels.append(graph.get_collection('joint_pred_labels')[i])
        attr_pred_labels = tf.concat(attr_total_pred_labels, axis=0)
        senti_pred_labels = tf.concat(senti_total_pred_labels, axis=0)
        joint_pred_labels = tf.concat(joint_total_pred_labels, axis=0)

        return attr_pred_labels, senti_pred_labels, joint_pred_labels

    def average_loss(self, graph, gpu_num):
        attr_total_loss = []
        senti_total_loss = []
        joint_total_loss = []
        for i in range(gpu_num):
            attr_total_loss.append(graph.get_collection('attr_loss')[i])
            senti_total_loss.append(graph.get_collection('senti_loss')[i])
            joint_total_loss.append(graph.get_collection('joint_loss')[i])
        attr_loss = tf.reduce_mean(attr_total_loss, axis=0)
        senti_loss = tf.reduce_mean(senti_total_loss,axis=0)
        joint_loss = tf.reduce_mean(joint_total_loss, axis=0)
        return attr_loss, senti_loss, joint_loss

    def compute_grads(self,mod,opt,tower_grads,graph):
        if mod == "attr":
            exception_list=['sentiExtr','table']
        else:
            exception_list = ['attrExtr','table']
        if self.nn_config['with_elmo']:
            exception_list.append('elmo')
            if mod == "attr" and 'table' not in exception_list:
                exception_list.append('table')
        print('exception list: ',exception_list)
        # all var
        vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # attribute
        var_list = []
        for var in vars:
            flag=True
            for expt in exception_list:
                if var.name.find(expt)>=0:
                    flag=False
                    break

            if flag:
                var_list.append(var)
        for var in var_list:
            print(var.name)
        print('==========================')

        grad = opt.compute_gradients(graph.get_collection(mod+'_loss')[-1],var_list=var_list,
                                     aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        if mod == "attr":
            tf.add_to_collection('attr_grads_and_vars',grad)
        tower_grads.append(grad)

    def build_models(self, Model):
        graph = tf.Graph()
        models = []
        attr_tower_grads = []
        senti_tower_grads = []
        joint_tower_grads = []
        with tf.device('/cpu:0'):
            with graph.as_default():
                global_step = tf.get_variable(
                    'global_step', [],
                    initializer=tf.constant_initializer(0), trainable=False)
                tf.add_to_collection('global_step',global_step)
                table = tf.placeholder(
                    shape=(self.nn_config['lookup_table_words_num'], self.nn_config['word_dim']),
                    dtype='float32')
                tf.add_to_collection('table',table)
                table = tf.Variable(table, name='table')
                tf.add_to_collection('var_table',table)
                opt = tf.train.AdamOptimizer(self.nn_config['lr'])
                for k in range(self.nn_config['gpu_num']):
                    with tf.device('/gpu:%d'%k):
                        with tf.variable_scope('sentiment', reuse=k > 0):
                            print('============ multiGPU %d ============' % k)
                            model = Model(self.nn_config, graph=graph,table=table)
                            models.append(model)

                            # attribute
                            self.compute_grads(mod = 'attr', opt=opt, tower_grads=attr_tower_grads, graph=graph)

                            # sentiment
                            self.compute_grads(mod = 'senti', opt=opt, tower_grads=senti_tower_grads,
                                               graph=graph)
                            # joint
                            self.compute_grads(mod = 'joint', opt=opt, tower_grads=joint_tower_grads,
                                               graph=graph)
                # gradient and train step
                attr_avg_grads = self.average_gradients(attr_tower_grads)
                # attr_clipped_grads, _ = self.clip_grads(attr_avg_grads,self.nn_config,False,global_step)
                attr_train_step = opt.apply_gradients(attr_avg_grads,global_step=global_step)

                senti_avg_grads = self.average_gradients(senti_tower_grads)
                # senti_clipped_grads,_ = self.clip_grads(senti_avg_grads,self.nn_config,False,global_step)
                senti_train_step = opt.apply_gradients(senti_avg_grads, global_step=global_step)

                joint_avg_grads = self.average_gradients(joint_tower_grads)
                # joint_clipped_grads, _ = self.clip_grads(joint_avg_grads, self.nn_config, False, global_step)
                joint_train_step = opt.apply_gradients(joint_avg_grads,global_step=global_step)
                # label
                attr_pred_labels, senti_pred_labels, joint_pred_labels = self.concat_pred_labels(graph, self.nn_config['gpu_num'])
                # loss
                attr_loss, senti_loss, joint_loss = self.average_loss(graph, self.nn_config['gpu_num'])
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        return {'train_step':{'attr':attr_train_step, 'senti':senti_train_step, 'joint':joint_train_step},
                'pred_labels':{'attr':attr_pred_labels, 'senti':senti_pred_labels, 'joint':joint_pred_labels},
                'loss':{'attr':attr_loss, 'senti':senti_loss, 'joint':joint_loss},
                'saver':saver,'graph':graph, 'gpu_num':self.nn_config['gpu_num'],'global_step':global_step}