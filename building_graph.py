import numpy as np
import scipy.sparse as sp

def train_test(df, test_rate = 0.1):
    test_ratio = test_rate
    test_idxs = []
    for b_id in df["Label"].unique():
        dum = df[df["Label"] == b_id]
        if len(dum) >= 4:
            test_idxs.extend(list(np.random.choice(dum.index, size=round(test_ratio * len(dum)), replace=False)))

    train_idxs = []
    for item in range(len(df["Label"])):
        if item not in test_idxs:
            train_idxs.append(item)

    return train_idxs,test_idxs

def word_frequency(df):
    # build vocab
    word_freq = {}
    word_set = set()
    for doc_words in df['Tokens']:
        for word in doc_words:
            word_set.add(word)
            if word in word_freq:
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    vocab = list(word_set)
    vocab_size = len(vocab)

    return word_freq, vocab, vocab_size

def word_doc(df):
    word_doc_list = {}
    for index in range(len(df['Tokens'])):
        appeared = set()
        words = df['Tokens'][index]
        for word in words:
            if word in appeared:
                continue
            if word in word_doc_list:
                doc_list = word_doc_list[word]
                doc_list.append(index)
                word_doc_list[word] = doc_list
            else:
                word_doc_list[word] = [index]
            appeared.add(word)

    return word_doc_list

def data_x_train_builder(df, word_vector_map, real_train_size, embed_dim):
    embed_dim = embed_dim
    row_x = []
    col_x = []
    data_x = []
    for i in range(real_train_size):
        doc_vec = np.array([0.0 for k in range(embed_dim)])
        words = df['Tokens'][i]
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                # print(doc_vec)
                # print(np.array(word_vector))
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(embed_dim):
            row_x.append(i)
            col_x.append(j)
            data_x.append(doc_vec[j] / doc_len)

    x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(real_train_size, embed_dim))
    return x

def data_y_train_builder(df, real_train_size):
    label_list = list(set(df['Label']))
    y = []
    for i in range(real_train_size):
        label = df['Label'][i]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        y.append(one_hot)
    y = np.array(y)
    return y, label_list

def data_x_test_builder(df, test_idxs, train_size, embed_dim, word_vector_map):
    test_size = len(test_idxs)

    row_tx = []
    col_tx = []
    data_tx = []
    for i in range(test_size):
        doc_vec = np.array([0.0 for k in range(embed_dim)])
        words = df['Tokens'][i + train_size]
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(embed_dim):
            row_tx.append(i)
            col_tx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

    tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),shape=(test_size, embed_dim))
    return tx, test_size

def data_y_test_builder(df, test_size, train_size, label_list):
    ty = []
    for i in range(test_size):
        label = df['Label'][i + train_size]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ty.append(one_hot)
    ty = np.array(ty)
    return ty

def data_allx(df, embed_dim, train_size, word_vector_map, vocab_size, word_vectors):
    row_allx = []
    col_allx = []
    data_allx = []

    for i in range(train_size):
        doc_vec = np.array([0.0 for k in range(embed_dim)])
        words = df['Tokens'][i]
        doc_len = len(words)
        for word in words:
            if word in word_vector_map:
                word_vector = word_vector_map[word]
                doc_vec = doc_vec + np.array(word_vector)

        for j in range(embed_dim):
            row_allx.append(int(i))
            col_allx.append(j)
            # np.random.uniform(-0.25, 0.25)
            data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
    for i in range(vocab_size):
        for j in range(embed_dim):
            row_allx.append(int(i + train_size))
            col_allx.append(j)
            data_allx.append(word_vectors.item((i, j)))

    row_allx = np.array(row_allx)
    col_allx = np.array(col_allx)
    data_allx = np.array(data_allx)
    allx = sp.csr_matrix((data_allx, (row_allx, col_allx)), shape=(train_size + vocab_size, embed_dim))

    return allx

def data_ally(df, train_size, label_list, vocab_size):
    ally = []
    for i in range(train_size):
        label = df['Label'][i]
        one_hot = [0 for l in range(len(label_list))]
        label_index = label_list.index(label)
        one_hot[label_index] = 1
        ally.append(one_hot)

    for i in range(vocab_size):
        one_hot = [0 for l in range(len(label_list))]
        ally.append(one_hot)

    ally = np.array(ally)
    return ally
