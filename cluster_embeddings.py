import pickle
import argparse
import os
from sklearn.cluster import HDBSCAN
import math
from tqdm import tqdm
import json


def cluster_embeddings_inner(embedding_dict):
    embedding_list = [value for values in embedding_dict.values() for value in values]

    # auto find optimal the hyperparameters and cluster
    if args.min_cluster_size is not None:
        hdb = HDBSCAN(min_cluster_size=args.min_cluster_size, n_jobs=-1, store_centers="medoid").fit(embedding_list)
        k = args.min_cluster_size
        print('num of clusters of', k, 'is', hdb.medoids_.shape[0])
    else:
        budget = args.relation_count * args.k_shot_per_relation
        min_cluster_size = math.ceil(len(embedding_list) / budget)
        print(min_cluster_size)
        for k in tqdm(range(2, min_cluster_size, 1)):
            hdb = HDBSCAN(min_cluster_size=k, n_jobs=-1, store_centers="medoid").fit(embedding_list)
            print('num of clusters of', k, 'is', hdb.medoids_.shape[0])
            if hdb.medoids_.shape[0] <= budget:
                break
    
    print('The clusters found: (cluster_num, embedding_dim): ', hdb.medoids_.shape)
    medoids = [list(embedding) for embedding in hdb.medoids_]
    return k, medoids


def cluster_embeddings(args):
    # read in embeddings and reformat
    embedding_dict = pickle.load(open(os.path.join(args.input_dir, args.input_file_name), 'rb'))
    k, medoids = cluster_embeddings_inner(embedding_dict)

    # find ids, sentence, entity of medoids as a list
    output_list = []
    # id_entity_data = json.load(open(os.path.join(args.input_dir, args.input_entities)))
    data = json.load(open(os.path.join('datasets598', args.dataset, 'preprocessed.json')))
    for id, sample2embeding in embedding_dict.items():
        for sample, embedding in sample2embeding.items():
            if list(embedding) in medoids: # list of embeddings (lists)
                obj = {'id': id, 'sentence': data['train'][id]['text'], 'entity/masked sentence': sample}
                output_list.append(obj)
                medoids.remove(embedding)
                break; # stop searching current id

    # print and save 
    print(output_list[0])
    print('Totally', args.relation_count, 'relations, with', args.k_shot_per_relation, 'shots per relation => picked min_cluster_size', k)
    json.dump(output_list, open(os.path.join(args.input_dir, args.output_file_name), 'w'))
    print('Write as a list of medoids embeddings.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='conll04')
    parser.add_argument('--input_dir', type=str, default='outputs/conll04_train')
    parser.add_argument('--input_embeddings', type=str, default='entity_embeddings.pickle') # senstence_embeddings.pickle
    parser.add_argument('--output_file_name', type=str, default='entity_medoids.json') # sentence_medoids.json

    parser.add_argument('--k_shot_per_relation', type=int, default=5)
    parser.add_argument('--refind_num_of_clusters', action='store_true')
    
    args = parser.parse_args()
    
    args.min_cluster_size = None
    if args.dataset == 'conll04':
        args.relation_count = 5
        args.min_cluster_size = 14 # exhaustive search
    elif args.dataset == 'ade':
        args.relation_count = 1
    elif args.dataset == 'nyt':
        args.relation_count = 24
    args.min_cluster_size = None if args.refind_num_of_clusters else args.min_cluster_size

    cluster_embeddings(args)
