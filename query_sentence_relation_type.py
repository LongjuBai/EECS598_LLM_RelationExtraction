from utils import make_icl_prompt
import json
import argparse
import pickle

sample_gt = json.load(open('outputs/conll04_train/sentence_medoids.json', 'r'))
embeddings = list(pickle.load(open('outputs/conll04_test/sentence_embeddings_test.pickle', 'rb'))['17'].values())
print(make_icl_prompt('conll04', sample_gt, embeddings, 10, mode='sentence'))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--dataset', type=str, default='conll04')
#     parser.add_argument('--input_dir', type=str, default='outputs/conll04_train')
#     parser.add_argument('--input_file_name', type=str, default='entity_embeddings.pickle') # senstence_embeddings.pickle
#     parser.add_argument('--output_file_name', type=str, default='entity_medoids.json') # sentence_medoids.json
#     parser.add_argument('--mode', type=str, default='entity') # mode or sentence

#     parser.add_argument('--k_shot_per_relation', type=int, default=5)
#     parser.add_argument('--refind_num_of_clusters', action='store_true')
    
#     args = parser.parse_args()
    
#     args.min_cluster_size = None
#     if args.dataset == 'conll04':
#         args.relation_count = 5
#         args.min_cluster_size = 14 if args.mode == 'entity' else 4 # exhaustive search
#     elif args.dataset == 'ade':
#         args.relation_count = 1
#     elif args.dataset == 'nyt':
#         args.relation_count = 24
#     args.min_cluster_size = None if args.refind_num_of_clusters else args.min_cluster_size

#     cluster_embeddings(args)
