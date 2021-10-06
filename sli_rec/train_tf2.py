import random
import numpy as np
import tensorflow as tf
from iterator import Iterator
import sys
from tqdm import tqdm
import math
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from model_tf2 import SLiRec_Adaptive

SEED = 3
MAX_EPOCH = 10
TEST_FREQ = 1000
LR = 1e-3
EMBEDDING_DIM = 18
HIDDEN_SIZE = 36
ATTENTION_SIZE = 36
MAXLEN = 100
DROPOUT = 0.0
l2_emb = 0.0
num_neg_test = 100
BATCH_SIZE = 128
MODEL_TYPE = "SLi_Rec_Adaptive"

MODEL_DICT = {
    # "ASVD": Model_ASVD,
    # "DIN": Model_DIN,
    # "LSTM": Model_LSTM,
    # "LSTMPP": Model_LSTMPP,
    # "NARM": Model_NARM,
    # "CARNN": Model_CARNN,  # baselines
    # "Time1LSTM": Model_Time1LSTM,
    # "Time2LSTM": Model_Time2LSTM,
    # "Time3LSTM": Model_Time3LSTM,
    # "DIEN": Model_DIEN,
    # "A2SVD": Model_A2SVD,
    # "T_SeqRec": Model_T_SeqRec,
    # "TC_SeqRec_I": Model_TC_SeqRec_I,
    # "TC_SeqRec_G": Model_TC_SeqRec_G,  # our models
    # "TC_SeqRec": Model_TC_SeqRec,
    # "SLi_Rec_Fixed": Model_SLi_Rec_Fixed,
    "SLi_Rec_Adaptive": SLiRec_Adaptive,
}


def calc_auc(raw_arr):
    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0.0, 0.0
    for record in arr:
        if record[1] == 1.0:
            pos += 1
        else:
            neg += 1

    fp, tp = 0.0, 0.0
    xy_arr = []
    for record in arr:
        if record[1] == 1.0:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.0
    prev_x = 0.0
    prev_y = 0.0
    for x, y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * (y + prev_y) / 2.0
            prev_x = x
            prev_y = y

    return auc


def evaluate_epoch(test_data, model):

    test_loss_sum = 0.0
    test_accuracy_sum = 0.0
    count = 0
    output = []
    y_true, y_score, y_pred = [], [], []
    y_true2 = []
    for src, tgt in test_data:
        count += 1
        all_inputs = prepare_data(src, tgt)
        inputs, label = create_dict(all_inputs)
        test_prob = model(inputs, training=False)
        test_loss = -tf.reduce_mean(tf.math.log(test_prob) * label)
        test_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.round(test_prob), label), tf.float32)
        )
        test_loss_sum += test_loss
        test_accuracy_sum += test_acc
        # print(test_prob[:10,:])
        # print(label[:10,:])
        test_prob_1 = test_prob[:, 0].numpy().tolist()
        label_1 = label[:, 0].tolist()
        pred_label = np.argmax(test_prob, axis=-1).tolist()
        true_label = np.argmax(label, axis=-1).tolist()
        # print(pred_label[:10])
        # print(true_label[:10])
        # sys.exit("HERE")
        y_pred.extend(pred_label)
        y_true2.extend(true_label)
        for p, t in zip(test_prob_1, label_1):
            output.append([p, t])
            y_true.append(t)
            y_score.append(p)

    test_auc = calc_auc(output)
    # y_pred = np.where(np.array(y_score)>0.5, 1, 0)
    print("ROC-AUC:", roc_auc_score(y_true, y_score), test_auc)
    print("precision:", precision_score(y_true2, y_pred))
    print("recall:", recall_score(y_true2, y_pred))
    test_loss = test_loss_sum / count
    test_accuracy = test_accuracy_sum / count

    return test_auc, test_loss, test_accuracy


def prepare_data(source, target, maxlen=MAXLEN):
    sequence_length = [len(s[3]) for s in source]
    item_history = [s[3] for s in source]
    cate_history = [s[4] for s in source]
    timeinterval_history = [s[5] for s in source]
    timelast_history = [s[6] for s in source]
    timenow_history = [s[7] for s in source]

    sequence_length1 = []
    item_history1 = []
    cate_history1 = []
    timeinterval_history1 = []
    timelast_history1 = []
    timenow_history1 = []
    for seqlen, inp in zip(sequence_length, source):
        if seqlen > maxlen:
            item_history1.append(inp[3][seqlen - maxlen :])
            cate_history1.append(inp[4][seqlen - maxlen :])
            timeinterval_history1.append(inp[5][seqlen - maxlen :])
            timelast_history1.append(inp[6][seqlen - maxlen :])
            timenow_history1.append(inp[7][seqlen - maxlen :])
            sequence_length1.append(maxlen)
        else:
            item_history1.append(inp[3])
            cate_history1.append(inp[4])
            timeinterval_history1.append(inp[5])
            timelast_history1.append(inp[6])
            timenow_history1.append(inp[7])
            sequence_length1.append(seqlen)

    sequence_length = sequence_length1
    item_history = item_history1
    cate_history = cate_history1
    timeinterval_history = timeinterval_history1
    timelast_history = timelast_history1
    timenow_history = timenow_history1

    if len(sequence_length) < 1:
        return None, None, None, None

    n_samples = len(item_history)
    maxlen_x = np.max(sequence_length)
    if maxlen_x < maxlen:
        maxlen_x = maxlen

    item_history_np = np.zeros((n_samples, maxlen_x)).astype("int64")
    cate_history_np = np.zeros((n_samples, maxlen_x)).astype("int64")
    timeinterval_history_np = np.zeros((n_samples, maxlen_x)).astype("float32")
    timelast_history_np = np.zeros((n_samples, maxlen_x)).astype("float32")
    timenow_history_np = np.zeros((n_samples, maxlen_x)).astype("float32")
    mid_mask = np.zeros((n_samples, maxlen_x)).astype("float32")
    for idx, [s_item, s_cate, s_tint, s_tlast, s_tnow] in enumerate(
        zip(
            item_history,
            cate_history,
            timeinterval_history,
            timelast_history,
            timenow_history,
        )
    ):
        mid_mask[idx, : sequence_length[idx]] = 1.0
        item_history_np[idx, : sequence_length[idx]] = s_item
        cate_history_np[idx, : sequence_length[idx]] = s_cate
        timeinterval_history_np[idx, : sequence_length[idx]] = s_tint
        timelast_history_np[idx, : sequence_length[idx]] = s_tlast
        timenow_history_np[idx, : sequence_length[idx]] = s_tnow

    user = np.array([inp[0] for inp in source])
    targetitem = np.array([inp[1] for inp in source])
    targetcategory = np.array([inp[2] for inp in source])

    return (
        user,
        targetitem,
        targetcategory,
        item_history_np,
        cate_history_np,
        timeinterval_history_np,
        timelast_history_np,
        timenow_history_np,
        mid_mask,
        np.array(target),
        np.array(sequence_length),
    )


def create_dict(tup):

    (
        user,
        targetitem,
        targetcategory,
        item_history,
        cate_history,
        timeinterval_history,
        timelast_history,
        timenow_history,
        mid_mask,
        label,
        seq_len,
    ) = tup
    inputs = {}
    inputs["users"] = np.expand_dims(np.array(user), axis=-1)
    inputs["targetitem"] = targetitem
    inputs["targetcategory"] = targetcategory
    inputs["item_history"] = item_history
    inputs["cate_history"] = cate_history
    # inputs["timeinterval_history"] = timeinterval_history
    inputs["timelast_history"] = timelast_history
    inputs["timenow_history"] = timenow_history
    inputs["mask"] = mid_mask

    return inputs, label


def train(
    train_file="data/train_data_reduced",
    test_file="data/test_data",
    save_path="saved_model/",
    model_type=MODEL_TYPE,
    seed=SEED,
):
    # tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if model_type in MODEL_DICT:
        cur_model = MODEL_DICT[model_type]
    else:
        print(f"{model_type} is not implemented")
        return

    train_data, test_data = Iterator(train_file), Iterator(test_file)
    user_number, item_number, cate_number = train_data.get_id_numbers()
    model = cur_model(
        item_num=item_number,
        cate_num=cate_number,
        seq_max_len=MAXLEN,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_SIZE,
        dropout_rate=DROPOUT,
        attention_dim=ATTENTION_SIZE,
        #    conv_dims = kwargs.get("conv_dims", [100, 100])
        l2_reg=l2_emb,
        num_neg_test=num_neg_test,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7
    )

    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.Mean(name="train_accuracy")

    train_step_signature = [
        {
            "users": tf.TensorSpec(shape=(None, 1), dtype=tf.int64),
            "targetitem": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            "targetcategory": tf.TensorSpec(shape=(None,), dtype=tf.int64),
            "item_history": tf.TensorSpec(shape=(None, MAXLEN), dtype=tf.int64),
            "cate_history": tf.TensorSpec(shape=(None, MAXLEN), dtype=tf.int64),
            "timelast_history": tf.TensorSpec(shape=(None, MAXLEN), dtype=tf.float32),
            "timenow_history": tf.TensorSpec(shape=(None, MAXLEN), dtype=tf.float32),
            "mask": tf.TensorSpec(shape=(None, MAXLEN), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32),
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, label):
        with tf.GradientTape() as tape:
            y_hat = model(inp, training=True)
            loss = -tf.reduce_mean(tf.math.log(y_hat) * label)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(y_hat), label), tf.float32))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy)
        return loss, accuracy

    itr = 0
    learning_rate = LR
    best_auc = 0.0
    best_model_path = save_path + model_type

    st = os.popen("wc -l data/train_data_reduced")
    num_train_data = int(st.read().split()[0])
    num_steps = math.ceil(num_train_data / BATCH_SIZE)

    print(
        "%g Users and %g items (%g categories)"
        % (user_number, item_number, cate_number)
    )

    for epoch in range(1, MAX_EPOCH + 1):
        train_loss_sum = 0.0
        train_accuracy_sum = 0.0
        step_loss = []
        train_loss.reset_states()

        # for step in tqdm(
        #     range(num_steps), total=num_steps, ncols=70, leave=False, unit="b"
        # ):
        #     src, tgt = next(train_data)
        # print(src)
        for src, tgt in train_data:

            all_inputs = prepare_data(src, tgt)
            # all_inputs has the following fields (in that order):
            # user, targetitem, targetcategory, item_history_np,
            # cate_history_np,
            # timeinterval_history_np, timelast_history_np, timenow_history_np,
            # mid_mask, np.array(target), np.array(sequence_length)

            inputs, target = create_dict(all_inputs)
            # for k in inputs:
            #     print(k, inputs[k].shape)
            # print("-----------------------")
            # print(inputs)
            # sys.exit("KK")

            loss, train_acc = train_step(inputs, target)
            # y_hat = model(inputs, training=False)
            # print(loss)
            step_loss.append(loss)
            train_loss_sum += loss
            train_accuracy_sum += train_acc
            itr += 1
            if (itr % TEST_FREQ) == 0:
                print(
                    f"Iter: {itr}, training loss = {train_loss_sum / TEST_FREQ}, training accuracy = {train_accuracy_sum / TEST_FREQ}"
                )

                test_auc, test_loss, test_acc = evaluate_epoch(test_data, model)
                print(
                    f"test_auc: {test_auc}, testing loss = {test_loss}, testing accuracy = {test_acc}"
                )

        print(
            f"Epoch: {epoch}, Train Loss: {np.mean(step_loss):.3f}, {train_loss.result():.3f}"
        )


if __name__ == "__main__":
    train()
