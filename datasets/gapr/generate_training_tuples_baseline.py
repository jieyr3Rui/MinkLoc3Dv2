# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pickle
import argparse
from tqdm import tqdm

from datasets.base_datasets import TrainingTuple


def construct_query_dict(base_path, filename):
    # Baseline dataset parameters in the original PointNetVLAD code: ind_nn_r=10, ind_r=50
    # Refined dataset parameters in the original PointNetVLAD code: ind_nn_r=12.5, ind_r=50
    all_items_name = os.listdir(os.path.join(base_path, "items"))
    all_items_name.sort()
    all_tums = np.load(os.path.join(base_path, "tum.npy"))
    queries = {}
    for item_name in tqdm(all_items_name):
        anchor_ndx = int(item_name)

        query = os.path.join(base_path, "items", item_name, "pointcloud.npy")

        timestamp = int(all_tums[anchor_ndx][0])

        # x, y
        anchor_pos = all_tums[anchor_ndx][1:3]

        positives = np.load(os.path.join(base_path, "items", item_name, "positives.npy"))
        non_negatives = np.load(os.path.join(base_path, "items", item_name, "non_negatives.npy"))

        # Sort ascending order
        positives = np.sort(positives)
        non_negatives = np.sort(non_negatives)

        # Tuple(id: int, timestamp: int, rel_scan_filepath: str, positives: List[int], non_negatives: List[int])
        queries[anchor_ndx] = TrainingTuple(
            id=anchor_ndx, 
            timestamp=timestamp, 
            rel_scan_filepath=query,
            positives=positives, 
            non_negatives=non_negatives, 
            position=anchor_pos
        )

    file_path = os.path.join(base_path, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    args = parser.parse_args()
    print('Dataset root: {}'.format(args.dataset_root))

    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root
    construct_query_dict(os.path.join(base_path, "train"),    "gapr_train.pickle")
    construct_query_dict(os.path.join(base_path, "evaluate"), "gapr_evaluate.pickle")
    