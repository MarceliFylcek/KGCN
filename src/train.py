import tensorflow as tf
import numpy as np
from model import KGCN
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


def train(args, data, show_loss, show_topk):
    n_user, n_item, n_entity, n_relation = data[0], data[1], data[2], data[3]
    train_data, eval_data, test_data = data[4], data[5], data[6]
    adj_entity, adj_relation = data[7], data[8]

    train_data = sort_by_time_stamp(train_data)
    eval_data = sort_by_time_stamp(eval_data) 
    test_data = sort_by_time_stamp(test_data)

    time_stamps = args.time_stamps

    # All items and relations passed
    model = KGCN(args, n_user, n_entity, n_relation, adj_entity, adj_relation)

    # top-K evaluation settings
    user_list, train_record, test_record, item_set, k_list = topk_settings(show_topk, train_data, test_data, n_item)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for step in range(args.n_epochs):
            
            shuffled_train_data = get_shuffled_copy(train_data)
            shuffled_eval_data = get_shuffled_copy(eval_data)
            shuffled_test_data = get_shuffled_copy(test_data)

            start = 0
            # skip the last incomplete minibatch if its size < batch size
            while start + args.batch_size <= shuffled_train_data.shape[0]:
                _, loss = model.train(sess, get_feed_dict(model, shuffled_train_data, train_data, time_stamps, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print(start, loss)

            # CTR evaluation
            train_auc, train_f1, _, _ = ctr_eval(sess, model, shuffled_train_data, train_data, time_stamps, args.batch_size)
            eval_auc, eval_f1, _, _ = ctr_eval(sess, model, shuffled_eval_data, eval_data, time_stamps, args.batch_size)
            test_auc, test_f1, labels, scores= ctr_eval(sess, model, shuffled_test_data, test_data, time_stamps, args.batch_size)
            # fpr, tpr, thresholds = roc_curve(y_true=labels, y_score=scores)

            # roc_auc = auc(fpr, tpr)

            # plt.figure()
            # plt.plot(fpr, tpr, color='darkorange', lw=2, label='Krzywa ROC (AUC = {:.2f})'.format(roc_auc))
            # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Losowy model')
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Krzywa ROC')
            # plt.show()

            print('epoch %d    train auc: %.4f  f1: %.4f    eval auc: %.4f  f1: %.4f    test auc: %.4f  f1: %.4f'
                 % (step, train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

            # # top-K evaluation
            if show_topk:
                precision, recall = topk_eval(
                    sess, model, user_list, train_record, test_record, item_set, k_list, args.batch_size)
                print('precision: ', end='')
                for i in precision:
                    print('%.4f\t' % i, end='')
                print()
                print('recall: ', end='')
                for i in recall:
                    print('%.4f\t' % i, end='')
                print('\n')


def topk_settings(show_topk, train_data, test_data, n_item):
    if show_topk:
        user_num = 100
        k_list = [1, 2, 5, 10, 20, 50, 100]
        train_record = get_user_record(train_data, True)
        test_record = get_user_record(test_data, False)
        user_list = list(set(train_record.keys()) & set(test_record.keys()))
        if len(user_list) > user_num:
            user_list = np.random.choice(user_list, size=user_num, replace=False)
        item_set = set(list(range(n_item)))
        return user_list, train_record, test_record, item_set, k_list
    else:
        return [None] * 5


def get_feed_dict(model: KGCN, data, data_sorted, time_stamps, start: int, end: int):
    history = get_history(data[start:end], data_sorted, time_stamps)

    feed_dict = {model.user_indices: data[start:end, 0],
                 model.item_history: history,
                 model.labels: data[start:end, 2]}
    return feed_dict

def get_history(data, data_sorted, n_time_stamps):
    histories = []
    for instance in data:
        history = []
        user_id = instance[0]
        time_stamp = instance[3]
        id = instance[4]

        # Last n_time_stamps
        if time_stamp != -1:
            last_existing_index = 0
            for i in range(n_time_stamps):
                # Check for out of bound index
                if id-i >= 0 and user_id == data_sorted[id-i][0]:
                    history.append(data_sorted[id-i][1])
                    last_existing_index = i
                # No more instances
                else:
                    history.append(data_sorted[id-last_existing_index][1])

        # No timestamp, get the newest products
        else:

            history.append(data_sorted[id][1])

            data_len = len(data_sorted)
            current_user = user_id
            current_index = id

            # Search for the newest product
            while(current_index < data_len-1):
                current_index += 1
                current_user = data_sorted[current_index][0]

                if current_user != user_id:
                    current_index -= 1
                    break

            last_existing_index = 0
            for i in range(n_time_stamps-1):
                # Check for out of bound index
                if id-i >= 0 and user_id == data_sorted[id-i][0]:
                    history.append(data_sorted[id-i][1])
                    last_existing_index = i
                # No more instances
                else:
                    history.append(data_sorted[id-last_existing_index][1])

        history.reverse()
        histories.append(history)
            

    final_history = np.stack(histories, axis=0)
    #history = np.expand_dims(data[start:end, 1], axis=0)
    # final_history = np.transpose(final_history)
    return final_history

def ctr_eval(sess, model, data, data_sorted, time_stamps, batch_size):
    start = 1
    auc_list = []
    f1_list = []
    labels_list = []
    scores_list = []
    while start + batch_size <= data.shape[0]:
        auc, f1, labels, scores = model.eval(sess, get_feed_dict(model, data, data_sorted, time_stamps, start, start + batch_size))
        auc_list.append(auc)
        f1_list.append(f1)
        labels_list.append(labels)
        scores_list.append(scores)
        start += batch_size
    labels_array = np.array(labels_list).flatten()
    scores_array = np.array(scores_list).flatten()

    return float(np.mean(auc_list)), float(np.mean(f1_list)), labels_array, scores_array


def topk_eval(sess, model, user_list, train_record, test_record, item_set, k_list, batch_size):
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}

    for user in user_list:
        test_item_list = list(item_set - train_record[user])
        item_score_map = dict()
        start = 0
        while start + batch_size <= len(test_item_list):
            items, scores = model.get_scores(sess, {model.user_indices: [user] * batch_size,
                                                    model.item_indices: test_item_list[start:start + batch_size]})
            for item, score in zip(items, scores):
                item_score_map[item] = score
            start += batch_size

        # padding the last incomplete minibatch if exists
        if start < len(test_item_list):
            items, scores = model.get_scores(
                sess, {model.user_indices: [user] * batch_size,
                       model.item_indices: test_item_list[start:] + [test_item_list[-1]] * (
                               batch_size - len(test_item_list) + start)})
            for item, score in zip(items, scores):
                item_score_map[item] = score

        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        for k in k_list:
            hit_num = len(set(item_sorted[:k]) & test_record[user])
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]

    return precision, recall


def get_user_record(data, is_train):
    user_history_dict = dict()
    for interaction in data:
        user = interaction[0]
        item = interaction[1]
        label = interaction[2]
        if is_train or label == 1:
            if user not in user_history_dict:
                user_history_dict[user] = set()
            user_history_dict[user].add(item)
    return user_history_dict

def sort_by_time_stamp(data):
    sorted_indices = np.lexsort((data[:, 3], data[:, 0]))
    sorted_data = data[sorted_indices]

    # Add index column
    indices = np.arange(len(sorted_data))
    sorted_data = np.column_stack([sorted_data, indices])

    return sorted_data

def get_shuffled_copy(data):
    n_instances = data.shape[0]
    random_indices = np.random.permutation(n_instances)
    shuffled_data = data[random_indices]

    return shuffled_data