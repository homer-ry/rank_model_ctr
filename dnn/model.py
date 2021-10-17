import tensorflow as tf

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell

class Model(object):

        
    def __init__(self, user_count, item_count, cate_count, cate_list):
        seed =1234
        tf.set_random_seed(seed)
        self.sess = tf.InteractiveSession()

        #build net
        self.u = tf.placeholder(tf.int32, [None,]) # [B]
        self.i = tf.placeholder(tf.int32, [None,]) # [B]
        self.j = tf.placeholder(tf.int32, [None,]) # [B]
        self.y = tf.placeholder(tf.float32, [None,]) # [B]
        self.hist_i = tf.placeholder(tf.int32, [None, None]) # [B, T]
        self.sl = tf.placeholder(tf.int32, [None,]) # [B]
        self.lr = tf.placeholder(tf.float64, [])

        hidden_units = 128

        user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
        item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
        item_b = tf.get_variable("item_b", [item_count],
                                                         initializer=tf.constant_initializer(0.0))
        cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
        cate_list = tf.convert_to_tensor(cate_list, dtype=tf.int64)

        u_emb = tf.nn.embedding_lookup(user_emb_w, self.u)
        print('u_emb shape=', u_emb.get_shape().as_list())
        

        ic = tf.gather(cate_list, self.i) #train获取cate类目切片
        i_emb = tf.concat(values = [
                tf.nn.embedding_lookup(item_emb_w, self.i),
                tf.nn.embedding_lookup(cate_emb_w, ic),
                ], axis=1)
        print('i_emb shape=', i_emb.get_shape().as_list())
        i_b = tf.gather(item_b, self.i)

        jc = tf.gather(cate_list, self.j) #test获取cate类目切片
        j_emb = tf.concat([
                tf.nn.embedding_lookup(item_emb_w, self.j),
                tf.nn.embedding_lookup(cate_emb_w, jc),
                ], axis=1)
        j_b = tf.gather(item_b, self.j)
        print('j_emb shape=', j_emb.get_shape().as_list())

        hc = tf.gather(cate_list, self.hist_i)
        print('cate_list=', cate_list)
        print('hc shape=', hc.get_shape().as_list())
        h_emb = tf.concat([
                tf.nn.embedding_lookup(item_emb_w, self.hist_i),
                tf.nn.embedding_lookup(cate_emb_w, hc),
                ], axis=2)
        print('h_emb shape=', h_emb.get_shape().as_list())

        #-- sum begin -------
        mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32) # [B, T]
        mask = tf.expand_dims(mask, -1) # [B, T, 1]
        mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]]) # [B, T, H]
        print('mask shape=', mask.get_shape().as_list())
        h_emb *= mask # [B, T, H]
        hist = h_emb
        hist = tf.reduce_sum(hist, 1) 
        hist = tf.div(hist, tf.cast(tf.tile(tf.expand_dims(self.sl,1), [1,128]), tf.float32))
        print( "h_emb shape:", h_emb.get_shape().as_list())
        #-- sum end ---------
        
        hist = tf.layers.batch_normalization(inputs = hist)
        hist = tf.reshape(hist, [-1, hidden_units])
        hist = tf.layers.dense(hist, hidden_units)

        u_emb = hist
        #-- fcn begin -------
        din_i = tf.concat([u_emb, i_emb], axis=-1)
        din_i = tf.layers.batch_normalization(inputs=din_i, name='b1')
        d_layer_1_i = tf.layers.dense(din_i, 80, activation=tf.nn.sigmoid, name='f1')
        d_layer_2_i = tf.layers.dense(d_layer_1_i, 40, activation=tf.nn.sigmoid, name='f2')
        d_layer_3_i = tf.layers.dense(d_layer_2_i, 1, activation=None, name='f3')
        din_j = tf.concat([u_emb, j_emb], axis=-1)
        din_j = tf.layers.batch_normalization(inputs=din_j, name='b1', reuse=True)
        d_layer_1_j = tf.layers.dense(din_j, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_j = tf.layers.dense(d_layer_1_j, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_j = tf.layers.dense(d_layer_2_j, 1, activation=None, name='f3', reuse=True)
        d_layer_3_i = tf.reshape(d_layer_3_i, [-1])
        d_layer_3_j = tf.reshape(d_layer_3_j, [-1])
        x = i_b - j_b + d_layer_3_i - d_layer_3_j # [B]
        self.logits = i_b + d_layer_3_i
        u_emb_all = tf.expand_dims(u_emb, 1)
        u_emb_all = tf.tile(u_emb_all, [1, item_count, 1])
        # logits for all item:
        all_emb = tf.concat([
                item_emb_w,
                tf.nn.embedding_lookup(cate_emb_w, cate_list)
                ], axis=1)
        all_emb = tf.expand_dims(all_emb, 0)
        all_emb = tf.tile(all_emb, [512, 1, 1])
        din_all = tf.concat([u_emb_all, all_emb], axis=-1)
        din_all = tf.layers.batch_normalization(inputs=din_all, name='b1', reuse=True)
        d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1', reuse=True)
        d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2', reuse=True)
        d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3', reuse=True)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, item_count])
        self.logits_all = tf.sigmoid(item_b + d_layer_3_all)
        #-- fcn end -------

        
        self.mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        self.score_i = tf.sigmoid(i_b + d_layer_3_i)
        self.score_j = tf.sigmoid(j_b + d_layer_3_j)
        self.score_i = tf.reshape(self.score_i, [-1, 1])
        self.score_j = tf.reshape(self.score_j, [-1, 1])
        self.p_and_n = tf.concat([self.score_i, self.score_j], axis=-1)
        print( self.p_and_n.get_shape().as_list())


        # Step variable
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.global_epoch_step = \
                tf.Variable(0, trainable=False, name='global_epoch_step')
        self.global_epoch_step_op = \
                tf.assign(self.global_epoch_step, self.global_epoch_step+1)

        self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=self.logits,
                        labels=self.y)
                )

        trainable_params = tf.trainable_variables()
        self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
        self.train_op = self.opt.apply_gradients(
                zip(clip_gradients, trainable_params), global_step=self.global_step)

        #initializes variables
        tf.global_variables_initializer().run()
        
        
#         # 4 梯度训练：choose gradient descent optimizer update step
#         self.global_step = tf.train.get_or_create_global_step()
#         optimizer = tf.train.AdamOptimizer(self.lr)
#         gradients, variables = zip(*optimizer.compute_gradients(self.loss))
#         clip_gradients, _ = tf.clip_by_global_norm(gradients, 5.)
#         self.opt = optimizer.apply_gradients(zip(clip_gradients, variables), global_step=self.global_step)
        
#         self.train_step = [self.optimizer, self.train_loss, self.p_true, self.p_hat, self.losses1, self.x, self.true_scores, self.p_true, self.vv]
#         self.eval_step = [self.win_agent, self.p_ratio, self.vv]
        
     

    def train(self, uij, l):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                self.u: uij[0], #[B] ,user list（1维特征）
                self.i: uij[1], #[B] ,item list（1维特征）
                self.y: uij[2], #[B] ,label list
                self.hist_i: uij[3], #[B, T] ,user seq list
                self.sl: uij[4], #[B] ,user seq len
                self.lr: l,
                })
#         print("[train debug]p_true:", len(p_true), len(p_true[0]), len(p_true[0][0]), p_true[0][0]) #[B, T, T]
#         print("[train debug]loss:", loss) #1
        return loss

    def eval(self, uij):
        u_auc, socre_p_and_n = self.sess.run([self.mf_auc, self.p_and_n], feed_dict={
                self.u: uij[0],
                self.i: uij[1],
                self.j: uij[2],
                self.hist_i: uij[3],
                self.sl: uij[4],
                })
        return u_auc, socre_p_and_n

    def test(self, uid, hist_i, sl):
        return self.sess.run(self.logits_all, feed_dict={
                self.u: uid,
                self.hist_i: hist_i,
                self.sl: sl,
                })

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path=path)

    def restore(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, save_path=path)

        
        
        
def extract_axis_1(data, ind):
    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)
    return res

