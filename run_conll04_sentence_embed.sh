# conll04 test
python3 generate_embeddings.py --test_k -1 --mode sentence --remove_entity_type --part test --path_to_entity_file outputs/conll04_test/output_conll04_seed=42_split=0_entity.json --path_to_sentence_file 'datasets598/conll04/preprocessed.json' --path_to_output_file outputs/conll04_test/sentence_embeddings_test.pickle

# conll04 train
python3 generate_embeddings.py --test_k -1 --mode sentence --remove_entity_type --part train --path_to_entity_file outputs/conll04_train/output_conll04_seed=42_split=0_entity.json --path_to_sentence_file 'datasets598/conll04/preprocessed.json' --path_to_output_file outputs/conll04_train/sentence_embeddings_train.pickle