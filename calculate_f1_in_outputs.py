import argparse
import os
import json
import numpy as np


def calculate_f1_in_outputs(dataset, n_shot):
    f1_list = []
    for i in range(1, 11):
        output = f'outputs/{dataset}_{n_shot}shot/output_{dataset}_seed=0_split={i}_sure.json'
        if os.path.exists(output):
            f1_list.append(json.load(open(output))['micro_f1'])
        else:
            break
    print(np.mean(f1_list)*100, np.std(f1_list)*100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='scierc')
    parser.add_argument('--n_shot', type=int, default=5)

    args = parser.parse_args()
    calculate_f1_in_outputs(args.dataset, args.n_shot)
