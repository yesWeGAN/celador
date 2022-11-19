# https://davidefiocco.github.io/nearest-neighbor-search-with-faiss/
import argparse
import gc
import os
from pathlib import Path

import faiss
import numpy as np
import torch
from faiss.contrib.ondisk import merge_ondisk
from tqdm import tqdm

import codebase.data.hashset
from codebase.data.dataset_structures import dataset_structures


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description='Train a KNN-index on given dataset.')
    parser.add_argument('-d', '--dataset', type=str, default="celador", help='dataset specification')
    args = parser.parse_args()
    return args

class KNNHashIndexTrainer:
    def __init__(self,
                 hashset: codebase.data.hashset.HashSet,
                 batchsize: int = 30000):
        self.hashset = hashset
        self.batchsize = batchsize
        self.outputpath = None

        self.embd_outputpath = Path(self.hashset._target_embd_knn_index_subdir)
        self.caption_outputpath = Path(self.hashset._target_cap_knn_index_subdir)
        self.combi_outputpath = Path(self.hashset._target_combi_knn_index_subdir)

        self.embd_vectors = None
        self.cap_vectors = None
        self.combi_vectors = None

        self.embd_index = None
        self.cap_index = None
        self.combi_index = None

        self.mode_triplets = {"embd": {"out": self.embd_outputpath,
                                       "vectorpath": os.path.join(self.hashset._target_embd_knn_tensor_subdir, "embd_vectors_combined.pt"),
                                       "index": self.embd_index},
                              "caption": {"out": self.caption_outputpath,
                                          "vectorpath": os.path.join(self.hashset._target_cap_knn_tensor_subdir, "caption_vectors_combined.pt"),
                                          "index": self.cap_index},
                              "combi": {"out": self.combi_outputpath,
                                        "vectorpath": os.path.join(self.hashset._target_combi_knn_tensor_subdir, "combivector.pt"),
                                        "index": self.combi_index}}
        print("Dumping hashset.")
        self.hashset = None
        gc.collect()

    def load_vectors(self, mode: str):
        self.mode_triplets[mode]["vectors"] = torch.load(self.mode_triplets[mode]["vectorpath"]).cpu().numpy()

    def train_index(self, mode: str):
        self.mode_triplets[mode]["index"].train(self.mode_triplets[mode]["vectors"][0:self.batchsize])

    def write_index(self, mode: str, filename="trained.index"):
        os.makedirs(self.mode_triplets[mode]["out"], exist_ok=True)
        faiss.write_index(self.mode_triplets[mode]["index"], os.path.join(self.mode_triplets[mode]["out"], filename))

    def read_index(self, mode: str, filename="trained.index"):
        self.mode_triplets[mode]["index"] = faiss.read_index(os.path.join(self.mode_triplets[mode]["out"], filename))

    def build_indexes(self):
        for mode in self.mode_triplets.keys():
            print(f"Loading vectors {mode}")
            self.load_vectors(mode)
            print(f"Building index {mode}")
            self.mode_triplets[mode]["index"] = faiss.index_factory(self.mode_triplets[mode]["vectors"].shape[1], "IVF400,Flat")

            self.train_index(mode=mode)
            self.write_index(mode=mode)

            n_batches = self.mode_triplets[mode]["vectors"].shape[0] // self.batchsize
            for i in tqdm(range(n_batches)):
                self.read_index(mode=mode)
                self.mode_triplets[mode]["index"].add_with_ids(
                    self.mode_triplets[mode]["vectors"][i * self.batchsize:(i + 1) * self.batchsize],
                    np.arange(i * self.batchsize, (i + 1) * self.batchsize))
                self.write_index(mode=mode, filename=f"block_{i}.index")
            self.read_index(mode=mode)

            block_fnames = [os.path.join(self.mode_triplets[mode]["out"], f"block_{b}.index") for b in range(n_batches)]
            print("Entering index merge mode..")
            merge_ondisk(self.mode_triplets[mode]["index"], block_fnames,
                         os.path.join(self.mode_triplets[mode]["out"], "merged_index.ivfdata"))
            self.write_index(mode=mode, filename="populated.index")

            print(f"Writing {mode} index complete. Cleaning up variables.")
            for block in block_fnames:
                os.remove(block)

            self.mode_triplets[mode]["index"] = None
            self.mode_triplets[mode]["vectors"] = None
            gc.collect()



def main():
    inputs = parse_args()
    dataset_path = os.path.join(dataset_structures[inputs.dataset]["target_dir"],
                                dataset_structures[inputs.dataset]["dataset_name"], "dataset")
    hashset = codebase.data.hashset.HashSet(dataset_path)

    hashindex = KNNHashIndexTrainer(hashset)
    hashindex.build_indexes()


if __name__ == '__main__':
    main()
