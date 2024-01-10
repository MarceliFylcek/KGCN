import argparse
import numpy as np
import os
import progressbar


RATING_FILE_NAME = dict({'movie': 'ratings.csv', 'book': 'BX-Book-Ratings.csv', 'music': 'user_artists.dat'})
SEP = dict({'movie': ',', 'book': ';', 'music': '\t'})
THRESHOLD = dict({'movie': 4, 'book': 0, 'music': 0})


def read_item_index_to_entity_id_file():
    file = '../data/' + DATASET + '/item_index2entity_id.txt'
    print('reading item index to entity id file: ' + file + ' ...')
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        satori_id = line.strip().split('\t')[1]
        item_index_old2new[item_index] = i
        entity_id2index[satori_id] = i
        i += 1


def convert_rating():
    input_file = '../data/' + DATASET + '/' + RATING_FILE_NAME[DATASET]
    output_file = '../data/' + DATASET + '/' + 'rating_timestamp_normalized.csv'

    if DATASET == "movie":
        normalize_timestamp(input_file, output_file)
        

    print('reading rating file ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings = dict()
    user_neg_ratings = dict()

    for line in open(output_file, encoding='utf-8').readlines()[1:]:
        array = line.strip().split(SEP[DATASET])

        if DATASET == 'movie':
            timestamp = array[3]
        # remove prefix and suffix quotation marks for BX dataset
        if DATASET == 'book':
            array = list(map(lambda x: x[1:-1], array))

        item_index_old = array[1]
        if item_index_old not in item_index_old2new:  # the item is not in the final item set
            continue
        item_index = item_index_old2new[item_index_old]

        user_index_old = int(array[0])

        rating = float(array[2])
        if rating >= THRESHOLD[DATASET]:
            if user_index_old not in user_pos_ratings:
                user_pos_ratings[user_index_old] = set()
            if DATASET == 'movie':
                user_pos_ratings[user_index_old].add((item_index, timestamp))
            else:
                user_pos_ratings[user_index_old].add(item_index)
        else:
            if user_index_old not in user_neg_ratings:
                user_neg_ratings[user_index_old] = set()
            if DATASET == 'movie':
                user_neg_ratings[user_index_old].add((item_index, timestamp))
            else:
                user_neg_ratings[user_index_old].add(item_index)

    print('converting rating file ...')
    writer = open('../data/' + DATASET + '/ratings_final.txt', 'w', encoding='utf-8')
    user_cnt = 0
    user_index_old2new = dict()
    for user_index_old, pos_item_set in user_pos_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]


        for item in pos_item_set:
            if DATASET == 'movie':
                item_id, timestamp = item
                writer.write('%d\t%d\t1\t%s\n' % (user_index, item_id, timestamp))
            else:
                writer.write('%d\t%d\t1\n' % (user_index, item))
        unwatched_set = item_set - pos_item_set

        if user_index_old in user_neg_ratings:
            user_neg_items = user_neg_ratings[user_index_old]
            neg_items_cnt = len(user_neg_items) # number of negative items
            pos_items_cnt = len(pos_item_set)

            number_of_items = pos_items_cnt if neg_items_cnt > pos_items_cnt else neg_items_cnt

            for i in range(number_of_items): # pick the same amount of negative records, random, if there are more negative records than positive
                rand = np.random.randint(low=0, high=len(user_neg_items))
                item = list(user_neg_items)[rand]

                if DATASET == 'movie':
                    item_id, timestamp = item
                    writer.write('%d\t%d\t0\t%s\n' % (user_index, item_id, timestamp))
                else:
                    writer.write('%d\t%d\t0\n' % (user_index, item))

            unwatched_set -= user_neg_ratings[user_index_old] # from all items set, remove the negative rating films
        
        for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
            # only the items which has not been interacted with by the user, are left
            if DATASET == 'movie':
                writer.write('%d\t%d\t0\t-1\n' % (user_index, item))
            else:
                writer.write('%d\t%d\t0\n' % (user_index, item))

    '''
    for user_index_old, neg_item_set in user_neg_ratings.items():
        if user_index_old not in user_index_old2new:
            user_index_old2new[user_index_old] = user_cnt
            user_cnt += 1
        user_index = user_index_old2new[user_index_old]

        neg_item_cnt = 0
        for item in neg_item_set:
            
            if DATASET == 'movie':
                item_id, timestamp = item
                writer.write('%d\t%d\t0\t%s\n' % (user_index, item_id, timestamp))
            else:
                writer.write('%d\t%d\t0\n' % (user_index, item))
            neg_item_cnt += 1
    '''

    writer.close()
    print('number of users: %d' % user_cnt)
    print('number of items: %d' % len(item_set))


def convert_kg():
    print('converting kg file ...')
    entity_cnt = len(entity_id2index)
    relation_cnt = 0

    writer = open('../data/' + DATASET + '/kg_final.txt', 'w', encoding='utf-8')
    for line in open('../data/' + DATASET + '/kg.txt', encoding='utf-8'):
        array = line.strip().split('\t')
        head_old = array[0]
        relation_old = array[1]
        tail_old = array[2]

        if head_old not in entity_id2index:
            entity_id2index[head_old] = entity_cnt
            entity_cnt += 1
        head = entity_id2index[head_old]

        if tail_old not in entity_id2index:
            entity_id2index[tail_old] = entity_cnt
            entity_cnt += 1
        tail = entity_id2index[tail_old]

        if relation_old not in relation_id2index:
            relation_id2index[relation_old] = relation_cnt
            relation_cnt += 1
        relation = relation_id2index[relation_old]

        writer.write('%d\t%d\t%d\n' % (head, relation, tail))

    writer.close()
    print('number of entities (containing items): %d' % entity_cnt)
    print('number of relations: %d' % relation_cnt)


def normalize_timestamp(input_file_path, output_file_path):    
    print("Reading lines from file.....")
    with open(input_file_path, encoding='utf-8') as file:
        lines = file.readlines()[1:]

        min_user_timestamps = {}  # Dictionary to store minimum timestamp for each user
        arrays = []

        lines_cnt = len(lines)

        for line in lines:
            array = line.strip().split(SEP[DATASET])
            userId = int(array[0])
            timestamp = int(array[3])

            if userId not in min_user_timestamps:
                min_user_timestamps[userId] = timestamp
            else:
                min_user_timestamps[userId] = min(min_user_timestamps[userId], timestamp)

            arrays.append(array)

        # Write normalized timestamps and other data
        os.system('cls')
        write_normalized_data(output_file_path, arrays, min_user_timestamps, lines_cnt)


def write_normalized_data(output_file, arrays, user_timestamps, lines_cnt):
    print("Lines to write:", lines_cnt)
    bar = progressbar.ProgressBar(maxval=lines_cnt, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    with open(output_file, 'a', encoding='utf-8') as writer:
        writer.write('userId,movieId,rating,timestamp\n')
        for i, array in enumerate(arrays):
            bar.update(i + 1)
            user_id = int(array[0])
            writer.write('%d,%d,%.1f,%d\n' % (user_id, int(array[1]), float(array[2]), int(array[3]) - user_timestamps[user_id]))
    bar.finish()
    os.system('cls')
    print("Finished normalizing the timestamp column")



if __name__ == '__main__':
    np.random.seed(555)

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, default='movie', help='which dataset to preprocess')
    args = parser.parse_args()
    DATASET = args.d

    entity_id2index = dict()
    relation_id2index = dict()
    item_index_old2new = dict()

    read_item_index_to_entity_id_file()
    convert_rating()
    convert_kg()

    print('done')