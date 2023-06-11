from preprocess_data import *
from building_graph import *
from word_doc_graph import *
from models import *

import gensim
import sys
import pickle as pkl
import scipy.sparse as sp
import numpy as np

import torch
from dgl import DGLGraph
import time
import datetime


def word2vec_model(df, embed_size):
    #building and training
    wd2vc_model = gensim.models.Word2Vec(df['Tokens'], min_count=1, vector_size=embed_size, window=5)
    wd2vc_model.train(df['Tokens'], epochs=10, total_examples=len(df['Tokens']))

    vocab = wd2vc_model.wv.key_to_index
    print("The total number of words are : ", len(vocab))
    vocab = list(vocab.keys())

    word_vector_map = {}
    for word in vocab:
        word_vector_map[word] = wd2vc_model.wv.get_vector(word)
    print("The no of key-value pairs : ", len(word_vector_map))  # should come equal to vocab size

    return word_vector_map

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def save_data_as_pickle(dataset_name, x_train, y_train, x_test, y_test, allx, ally, adj):

    # dump objects
    dataset = dataset_name
    with open("./data_pickles/ind.{}.x".format(dataset), 'wb') as f:
        pkl.dump(x_train, f)

    with open("./data_pickles/ind.{}.y".format(dataset), 'wb') as f:
        pkl.dump(y_train, f)

    with open("./data_pickles/ind.{}.tx".format(dataset), 'wb') as f:
        pkl.dump(x_test, f)

    with open("./data_pickles/ind.{}.ty".format(dataset), 'wb') as f:
        pkl.dump(y_test, f)

    with open("./data_pickles/ind.{}.allx".format(dataset), 'wb') as f:
        pkl.dump(allx, f)

    with open("./data_pickles/ind.{}.ally".format(dataset), 'wb') as f:
        pkl.dump(ally, f)

    with open("./data_pickles/ind.{}.adj".format(dataset), 'wb') as f:
        pkl.dump(adj, f)

def load_corpus(dataset_str):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("./data_pickles/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    # print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    # print(len(labels))

    train_idx_orig = parse_index_file("./data_pickles/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # return sparse_to_tuple(features)
    return features.A

def pre_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def construct_graph(adjacency):
    g = DGLGraph()
    adj = pre_adj(adjacency)
    g.add_nodes(adj.shape[0])
    g.add_edges(adj.row, adj.col)
    adjdense = adj.A
    adjd = np.ones((adj.shape[0]))
    for i in range(adj.shape[0]):
        adjd[i] = adjd[i] * np.sum(adjdense[i, :])
    weight = torch.from_numpy(adj.data.astype(np.float32))
    g.ndata['d'] = torch.from_numpy(adjd.astype(np.float32))
    g.edata['w'] = weight

    return g

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized.A

import argparse

def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=bool, default=False,
                        help='Use CUDA training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.02,
                        help='Initial learning rate.')
    parser.add_argument('--model', type=str, default="GCN",
                        choices=["GCN", "SAGE", "GAT"],
                        help='model to use.')
    parser.add_argument('--early_stopping', type=int, default=10,
                        help='require early stopping.')

    args, _ = parser.parse_known_args()
    #args.cuda = not args.no_cuda and th.cuda.is_available()
    return args

def evaluate(features, labels, mask):
    t_test = time.time()
    model.eval()
    with torch.no_grad():
        logits = model(features).cpu()
        t_mask = torch.from_numpy(np.array(mask * 1., dtype=np.float32))
        tm_mask = torch.transpose(torch.unsqueeze(t_mask, 0), 1, 0).repeat(1, labels.shape[1])
        loss = criterion(logits * tm_mask, torch.max(labels, 1)[1])
        pred = torch.max(logits, 1)[1]
        acc = ((pred == torch.max(labels, 1)[1]).float() * t_mask).sum().item() / t_mask.sum().item()

    return loss.numpy(), acc, pred.numpy(), labels.numpy(), (time.time() - t_test)

def print_log(msg='', end='\n'):
    now = datetime.datetime.now()
    t = str(now.year) + '/' + str(now.month) + '/' + str(now.day) + ' ' \
        + str(now.hour).zfill(2) + ':' + str(now.minute).zfill(2) + ':' + str(now.second).zfill(2)

    if isinstance(msg, str):
        lines = msg.split('\n')
    else:
        lines = [msg]

    for line in lines:
        if line == lines[-1]:
            print('[' + t + '] ' + str(line), end=end)
        else:
            print('[' + t + '] ' + str(line))


if __name__ == '__main__':
    #Reading data

    dataset = 'Reddit_SD'
    # dataset = 'Reddit_SNS'
    length = 5000
    embed_dim = 300
    window_size = 20

    if dataset == 'Twitter_10000':
        #twitter 10000
        df = read_twitter_10000(length)
    elif dataset == 'Twitter_tendency':
        #twitter_tendency
        df = read_twitter_tendency(length)
    elif dataset == 'Reddit_SNS':
        #reddit_sns
        df = read_reddit_SNS(length)
    elif dataset == 'Reddit_SD':
        #reddit_sd
        df = read_reddit_SD(length)

    print(df)
    dataset_name = dataset + str(len(df['Tokens']))

    word_vector_map = word2vec_model(df, embed_size=embed_dim)
    train_idxs, test_idxs = train_test(df, 0.1)

    train_ids_str = '\n'.join(str(index) for index in train_idxs)
    with open('./data_pickles/' + dataset_name + '.train.index', 'w') as f:
        f.write(train_ids_str)

    word_freq, vocab, vocab_size = word_frequency(df)
    word_doc_list = word_doc(df)

    word_doc_freq = {}
    for word, doc_list in word_doc_list.items():
        word_doc_freq[word] = len(doc_list)

    word_id_map = {}
    for i in range(vocab_size):
        word_id_map[vocab[i]] = i

    train_size = len(train_idxs)
    val_size = int(0.1 * train_size)
    real_train_size = train_size - val_size  # - int(0.5 * train_size)
    print(train_size, val_size, real_train_size)

    x_train = data_x_train_builder(df, word_vector_map, real_train_size, embed_dim)
    y_train, label_list = data_y_train_builder(df, real_train_size)

    x_test, test_size = data_x_test_builder(df, test_idxs, train_size, embed_dim, word_vector_map)
    y_test = data_y_test_builder(df, test_size, train_size, label_list)

    word_vectors = np.random.uniform(-0.01, 0.01, (vocab_size, embed_dim))

    for i in range(len(vocab)):
        word = vocab[i]
        if word in word_vector_map:
            vector = word_vector_map[word]
            word_vectors[i] = vector

    allx = data_allx(df, embed_dim, train_size, word_vector_map, vocab_size, word_vectors)
    ally = data_ally(df, train_size, label_list, vocab_size)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape, allx.shape, ally.shape)

    windows = word_coccurrance(df, window_size)
    word_window_freq = windows_frequency(windows)
    word_pair_count = words_pair_count(windows, word_id_map)

    row, col, weight = pmi_calculator(windows, word_pair_count, word_window_freq, vocab, train_size)
    doc_word_freq = doc_word_frequency(df, word_id_map, train_size, row, vocab_size, col, weight, word_doc_freq, vocab)

    node_size = train_size + vocab_size + test_size
    adj = sp.csr_matrix((weight, (row, col)), shape=(node_size, node_size))

    save_data_as_pickle(dataset_name, x_train, y_train, x_test, y_test, allx, ally, adj)

    ######################################### Enf of constructing the associated graph

    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus(dataset_name)

    features = sp.identity(features.shape[0])
    features = preprocess_features(features)

    adjdense = torch.from_numpy(pre_adj(adj).A.astype(np.float32))

    g = construct_graph(adj)

    # Define placeholders
    t_features = torch.from_numpy(features.astype(np.float32))
    t_y_train = torch.from_numpy(y_train)
    t_y_val = torch.from_numpy(y_val)
    t_y_test = torch.from_numpy(y_test)
    t_train_mask = torch.from_numpy(train_mask.astype(np.float32))
    tm_train_mask = torch.transpose(torch.unsqueeze(t_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
    support = [preprocess_adj(adj)]
    num_supports = 1
    t_support = []
    for i in range(len(support)):
        t_support.append(torch.Tensor(support[i]))

    args = get_citation_args()

    ## Train
    # GCN
    model1 = Classifer(g, input_dim=features.shape[0], num_classes=y_train.shape[1], conv=SimpleConv)
    # SAGE
    model2 = Classifer(g, input_dim=features.shape[0], num_classes=y_train.shape[1], conv=SAGEMeanConv)
    # GAT
    model3 = Classifer(g, input_dim=features.shape[0], num_classes=y_train.shape[1], conv=MultiHeadGATLayer)

    print(model1, '\n--\n', model2, '\n--\n', model3)

    model = model3

    # Loss and optimizer
    lr = 0.05
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    val_losses = []
    ###################################################
    ## Training Phase
    # Train the built model
    for epoch in range(args.epochs):

        t = time.time()

        # Forward pass
        logits = model(t_features)
        loss = criterion(logits * tm_train_mask, torch.max(t_y_train, 1)[1])
        acc = ((torch.max(logits, 1)[1] == torch.max(t_y_train, 1)[
            1]).float() * t_train_mask).sum().item() / t_train_mask.sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        val_loss, val_acc, pred, labels, duration = evaluate(t_features, t_y_val, val_mask)
        val_losses.append(val_loss)

        print_log(
            "Epoch: {:.0f}, train_loss= {:.5f}, train_acc= {:.5f}, val_loss= {:.5f}, val_acc= {:.5f}, time= {:.5f}" \
            .format(epoch + 1, loss, acc, val_loss, val_acc, time.time() - t))

        if epoch > args.early_stopping and val_losses[-1] > np.mean(val_losses[-(args.early_stopping + 1):-1]):
            print_log("Early stopping...")
            break

    print_log("Optimization Finished!")

    import sklearn.metrics as metrics

    ################################################################
    # Test phase
    # Testing the trained model
    test_loss, test_acc, pred, labels, test_duration = evaluate(t_features, t_y_test, test_mask)
    print_log("Test set results: \n\t loss= {:.5f}, accuracy= {:.5f}, time= {:.5f}".format(test_loss, test_acc,
                                                                                           test_duration))

    test_pred = []
    test_labels = []
    for i in range(len(test_mask)):
        if test_mask[i]:
            test_pred.append(pred[i])
            test_labels.append(np.argmax(labels[i]))

    print_log("Test Precision, Recall and F1-Score...")
    print_log(metrics.classification_report(test_labels, test_pred, digits=4))
    print_log("Macro average Test Precision, Recall and F1-Score...")
    print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
    print_log("Micro average Test Precision, Recall and F1-Score...")
    print_log(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))


    # doc and word embeddings
    tmp = model.embedding.cpu().numpy()
    word_embeddings = tmp[train_size: adj.shape[0] - test_size]
    train_doc_embeddings = tmp[:train_size]  # include val docs
    test_doc_embeddings = tmp[adj.shape[0] - test_size:]

    print_log('Embeddings:')
    print_log('\rWord_embeddings:'+str(len(word_embeddings)))
    print_log('\rTrain_doc_embeddings:'+str(len(train_doc_embeddings)))
    print_log('\rTest_doc_embeddings:'+str(len(test_doc_embeddings)))
    print_log('\rWord_embeddings:')
    print(word_embeddings)

