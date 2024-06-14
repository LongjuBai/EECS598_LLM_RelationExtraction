# conll04 test
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_test
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_test --remove_entity_type
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_test --mode sentence --part test
python3 generate_embeddings.py --test_k -1 --mode entity --remove_entity_type --part test --path_to_entity_file outputs/conll04_test/output_conll04_seed=42_split=0_entity.json --path_to_output_file outputs/conll04_test/entity_embeddings_test.pickle

# conllo4 train
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_train
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_train --remove_entity_type
# python3 generate_enbeddings.py --entity_file_name output_conll04_seed=42_split=0_entity.json --dataset conll04 --test_k -1 --suffix CONLL04_train --mode sentence --part train
python3 generate_embeddings.py --test_k -1 --mode entity --remove_entity_type --part train --path_to_entity_file outputs/conll04_train/output_conll04_seed=42_split=0_entity.json --path_to_output_file outputs//entity_embeddings_train.pickle