import tensorflow as tf
import numpy as np
import sys
from talstm import TALSTMCell


class attention_FCN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(attention_FCN, self).__init__()

        self.attention_size = kwargs.get("attention_size", None)
        self.stag = kwargs.get("stag", "null")
        self.mode = kwargs.get("mode", "SUM")
        self.softmax_stag = kwargs.get("softmax_stag", 1)
        self.time_major = kwargs.get("time_major", False)
        self.return_alphas = kwargs.get("return_alphas", False)
        self.forCNN = kwargs.get("forCNN", False)

        self.facts_size = kwargs.get("facts_size", False)
        self.querry_size = kwargs.get("querry_size", False)
        self.seq_len = kwargs.get("sequence_length", None)

        self.prelu = tf.keras.layers.PReLU(
            alpha_initializer="zeros",
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=None,
        )

        self.mlp1 = tf.keras.layers.Dense(units=self.facts_size, activation=None)
        self.mlp2 = tf.keras.layers.Dense(units=80, activation="sigmoid")
        self.mlp3 = tf.keras.layers.Dense(units=40, activation="sigmoid")
        self.mlp4 = tf.keras.layers.Dense(units=1, activation=None)

    def call(self, query, facts, mask):
        # query = target_item_embedding
        # facts = rnn_output, (B, S, H)
        # Trainable parameters
        mask = tf.equal(mask, tf.ones_like(mask))
        # facts_size = facts.shape[-1]  # D value - hidden size of the RNN layer
        # querry_size = query.shape[-1]
        query = self.mlp1(query)
        query = self.prelu(query)  # (None, 2, H)

        # print("query:", query.shape, self.seq_len)

        queries = tf.tile(query, [1, self.seq_len])
        # print("queries:", queries.shape)
        queries = tf.reshape(queries, [-1, self.seq_len, self.facts_size])
        din_all = tf.concat(
            [queries, facts, queries - facts, queries * facts], axis=-1
        )  # (128, 100, 36 * 4)

        d_layer_1_all = self.mlp2(din_all)
        d_layer_2_all = self.mlp3(d_layer_1_all)
        d_layer_3_all = self.mlp4(d_layer_2_all)
        d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, self.seq_len])
        scores = d_layer_3_all

        # Mask
        key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-(2 ** 32) + 1)
        if not self.forCNN:
            scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

        # Activation
        if self.softmax_stag:
            scores = tf.nn.softmax(scores)  # [B, 1, T]

        # Weighted sum
        if self.mode == "SUM":
            output = tf.matmul(scores, facts)  # [B, 1, H]
        else:
            scores = tf.reshape(scores, [-1, self.seq_len])
            output = facts * tf.expand_dims(scores, -1)
            output = tf.reshape(output, tf.shape(facts))
        if self.return_alphas:
            return output, scores
        return output


class attention_HAN(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(attention_HAN, self).__init__()

        self.attention_size = kwargs.get("attention_size", None)
        self.return_alphas = kwargs.get("return_alphas", False)
        self.seq_len = kwargs.get("sequence_length", None)
        self.embedding_dim = kwargs.get("embedding_dim", None)

    def build(self, input_shape):
        # print("input_shape", input_shape)
        hidden_size = input_shape[2]
        if not self.attention_size:
            self.attention_size = hidden_size

        self.kernel = self.add_weight(
            shape=(hidden_size, self.attention_size),
            name="w_omega",
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.attention_size * 2,),
            name="b_omega",
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1),
            trainable=True,
        )

    def call(self, x):
        v = tf.tanh(
            tf.tensordot(x, self.kernel, axes=1) + self.bias[: self.attention_size]
        )
        vu = tf.tensordot(
            v, self.bias[self.attention_size :], axes=1, name="vu"
        )  # (B,T) shape
        alphas = tf.nn.softmax(vu, name="alphas")  # (B,T) shape
        # print("alphas", alphas.shape)
        alphas = tf.reshape(alphas, [-1, x.shape[1]])
        # print("alphas", alphas.shape)
        output = x * tf.expand_dims(alphas, -1)
        # print("output", output.shape, self.seq_len, self.embedding_dim)
        output = tf.reshape(output, [-1, self.seq_len, self.embedding_dim])
        # print("output", output.shape)

        if not self.return_alphas:
            return output
        else:
            return output, alphas

    def fcn_net(self, inps, use_dice=False):
        bn1 = tf.layers.batch_normalization(inputs=inps, name="bn1")
        dnn1 = tf.layers.dense(bn1, 200, activation=None, name="f1")
        if use_dice:
            dnn1 = dice(dnn1, name="dice_1")
        else:
            dnn1 = prelu(dnn1, "prelu1")

        dnn2 = tf.layers.dense(dnn1, 80, activation=None, name="f2")
        if use_dice:
            dnn2 = dice(dnn2, name="dice_2")
        else:
            dnn2 = prelu(dnn2, "prelu2")
        dnn3 = tf.layers.dense(dnn2, 2, activation=None, name="f3")
        self.y_hat = tf.nn.softmax(dnn3) + 0.00000001

        with tf.name_scope("Metrics"):
            ctr_loss = -tf.reduce_mean(tf.log(self.y_hat) * self.label)
            self.loss = ctr_loss
            tf.summary.scalar("loss", self.loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                self.loss
            )
            self.accuracy = tf.reduce_mean(
                tf.cast(tf.equal(tf.round(self.y_hat), self.label), tf.float32)
            )
            tf.summary.scalar("accuracy", self.accuracy)

        self.merged = tf.summary.merge_all()


class SLiRec_Adaptive(tf.keras.Model):
    """SliRec model
    Self-Attentive Sequential Recommendation Using Transformer

    :Citation:

        Wang-Cheng Kang, Julian McAuley (2018), Self-Attentive Sequential
        Recommendation. Proceedings of IEEE International Conference on
        Data Mining (ICDM'18)

        Original source code from nnkkmto/SASRec-tf2, https://github.com/nnkkmto/SASRec-tf2

    Args:
        item_num: number of items in the dataset
        seq_max_len: maximum number of items in user history
        num_blocks: number of Transformer blocks to be used
        embedding_dim: item embedding dimension
        attention_dim: Transformer attention dimension
        conv_dims: list of the dimensions of the Feedforward layer
        dropout_rate: dropout rate
        l2_reg: coefficient of the L2 regularization
        num_neg_test: number of negative examples used in testing
    """

    def __init__(self, **kwargs):
        super(SLiRec_Adaptive, self).__init__()

        self.item_num = kwargs.get("item_num", None)
        self.cate_num = kwargs.get("cate_num", None)
        self.seq_max_len = kwargs.get("seq_max_len", 100)
        self.embedding_dim = kwargs.get("embedding_dim", 100)
        self.hidden_dim = kwargs.get("hidden_dim", 100)
        self.attention_dim = kwargs.get("attention_dim", 100)
        self.conv_dims = kwargs.get("conv_dims", [100, 100])
        self.dropout_rate = kwargs.get("dropout_rate", 0.5)
        self.l2_reg = kwargs.get("l2_reg", 0.0)
        self.num_neg_test = kwargs.get("num_neg_test", 100)

        print(tf.__version__)

        self.item_embedding_layer = tf.keras.layers.Embedding(
            self.item_num + 1,
            self.embedding_dim,
            name="item_embeddings",
            mask_zero=True,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )  # self.item_lookup
        self.category_embedding_layer = tf.keras.layers.Embedding(
            self.cate_num + 1,
            self.embedding_dim,
            name="category_embeddings",
            mask_zero=True,
            embeddings_regularizer=tf.keras.regularizers.L2(self.l2_reg),
        )  # self.cate_lookup

        self.attention1 = attention_HAN(
            attention_size=self.attention_dim,
            return_alphas=True,
            sequence_length=self.seq_max_len,
            embedding_dim=self.embedding_dim * 2,  # item + category embeddings
        )

        self.attention2 = attention_FCN(
            attention_size=self.attention_dim,
            softmax_stag=1,
            stag="1_1",
            mode="LIST",
            return_alphas=True,
            facts_size=self.hidden_dim,  # RNN output
            querry_size=self.embedding_dim * 2,  # item + category embeddings
            sequence_length=self.seq_max_len,
        )

        # Time-Aware LSTM
        self.rnn = tf.keras.layers.RNN(
            TALSTMCell(self.hidden_dim),
            input_shape=(None, self.embedding_dim + 2),
            return_sequences=True,
            return_state=True,
        )

        self.mlp1 = tf.keras.layers.Dense(units=80, activation="sigmoid")
        self.mlp2 = tf.keras.layers.Dense(units=40, activation="sigmoid")
        self.mlp3 = tf.keras.layers.Dense(units=1, activation="sigmoid")

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dnn1 = tf.keras.layers.Dense(units=200, activation=None)
        self.prelu1 = tf.keras.layers.PReLU(
            alpha_initializer="zeros",
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=None,
        )
        self.dnn2 = tf.keras.layers.Dense(units=80, activation=None)
        self.prelu2 = tf.keras.layers.PReLU(
            alpha_initializer="zeros",
            alpha_regularizer=None,
            alpha_constraint=None,
            shared_axes=None,
        )
        self.dnn3 = tf.keras.layers.Dense(units=2, activation=None)

    def call(self, x, training):

        targetitem = x["targetitem"]
        targetcategory = x["targetcategory"]
        item_history = x["item_history"]
        cate_history = x["cate_history"]
        timelast_history = x["timelast_history"]
        timenow_history = x["timenow_history"]
        mask = x["mask"]

        # print(targetitem)
        # print(targetcategory)

        targetitem_embedding = self.item_embedding_layer(targetitem)
        itemhistory_embedding = self.item_embedding_layer(item_history)

        targetcate_embedding = self.category_embedding_layer(targetcategory)
        catehistory_embedding = self.category_embedding_layer(cate_history)

        # print("target:", targetitem.shape, targetcategory.shape)
        # print("target", targetitem_embedding.shape, targetcate_embedding.shape)

        target_item_embedding = tf.concat(
            [targetitem_embedding, targetcate_embedding], 1
        )

        # print(target_item_embedding)

        item_history_embedding = tf.concat(
            [itemhistory_embedding, catehistory_embedding], 2
        )

        # print("item_history_embedding", item_history_embedding.shape)  # (b, s, h)
        # Attention Layer - I
        att_outputs1, alphas1 = self.attention1(item_history_embedding)  # (b, s, h)
        # print(att_outputs1.shape)
        att_fea1 = tf.reduce_sum(att_outputs1, 1)
        # print(att_fea1.shape)
        # sys.exit()

        item_history_embedding_new = tf.concat(
            [item_history_embedding, tf.expand_dims(timelast_history, -1)], -1
        )
        item_history_embedding_new = tf.concat(
            [item_history_embedding_new, tf.expand_dims(timenow_history, -1)], -1
        )

        # print(item_history_embedding_new.shape)  # (b, s, h+2)
        outputs = self.rnn(item_history_embedding_new)
        rnn_outputs = outputs[0]
        # final_state = outputs[1:]

        # print("K1", target_item_embedding.shape, rnn_outputs.shape)

        # Attention Layer - II
        att_outputs2, alphas2 = self.attention2(
            target_item_embedding, rnn_outputs, mask
        )
        # print("HERE: ", att_outputs2.shape, alphas2.shape)  # (128, 100, 36), (128, 100)

        att_fea2 = tf.reduce_sum(att_outputs2, 1)

        # print(
        #     target_item_embedding.shape,
        #     att_fea1.shape,
        #     att_fea2.shape,
        #     timenow_history.shape,
        # )

        concat_all = tf.concat(
            [
                target_item_embedding,
                att_fea1,
                att_fea2,
                tf.expand_dims(timenow_history[:, -1], -1),
            ],
            1,
        )

        concat_att1 = self.mlp1(concat_all)
        concat_att2 = self.mlp2(concat_att1)
        user_alpha = self.mlp3(concat_att2)
        user_embed = att_fea1 * user_alpha + att_fea2 * (1.0 - user_alpha)

        # print("HERE-2", target_item_embedding.shape, user_embed.shape)

        last_inps = tf.concat([target_item_embedding, user_embed], 1)  # (B, 2*H)
        bn1 = self.bn1(last_inps)
        bn1 = self.dnn1(bn1)
        bn1 = self.prelu1(bn1)  # need to change to dice
        bn2 = self.dnn2(bn1)
        bn2 = self.prelu2(bn2)
        bn2 = self.dnn3(bn2)

        y_hat = tf.nn.softmax(bn2) + 0.00000001  # (B,2)

        return y_hat
