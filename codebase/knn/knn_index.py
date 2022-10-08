# https://davidefiocco.github.io/nearest-neighbor-search-with-faiss/
import argparse
import json
import os
import shutil
from pathlib import Path
import faiss
import numpy as np
import torch
from faiss.contrib.ondisk import merge_ondisk

from codebase.knn.embedder import Embedder


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description='Embed a folder of images.')
    parser.add_argument('-i', '--inputpath', type=str, help='folder to process')
    parser.add_argument('-o', '--outputpath', type=str, help='folder for output')
    parser.add_argument('-bs', '--batchsize', type=int, help='batchsize for embedder model')

    args = parser.parse_args()
    kwargs = vars(args)  # convert namespace to dictionary
    return kwargs


class KNNIndexInference:
    def __init__(self, inputpath, dataset, outputpath=None, batchsize=100):
        self.inputpath = Path(inputpath)
        if outputpath:
            self.outputpath = Path(outputpath)
        else:
            self.outputpath = os.path.join(self.inputpath, "knn_result")
        self.batchsize = batchsize
        self.dataset = dataset
        self.index = self.find_indexfile()

    def find_indexfile(self):
        try:
            indexfile = next(Path(self.dataset).rglob("populated.index"))
            return faiss.read_index(indexfile.as_posix())
        except StopIteration:
            raise FileNotFoundError("No index file found. Exiting.")

    def find_jsonfile(self):
        try:
            filep = next(Path(self.dataset).rglob("**/filepath_index.json"))
            return json.load(open(filep, 'r'))
        except StopIteration:
            raise FileNotFoundError("No json file found. Exiting.")

    def embed_query(self):
        embedder = Embedder(inputpath=self.inputpath, outputpath=self.outputpath, batchsize=self.batchsize)
        embedder.embed()
        embedder.process_output()  # by the end of this, there will be a tensors.pt file in self.inputpath

    def search_full_index(self, vectors, k):
        self.index.nprobe = 80
        distances, neighbors = self.index.search(vectors, k)
        return distances, neighbors

    def find_tensors_and_convert(self):
        tensorpath = next(Path(self.inputpath).rglob("**/tensors.pt"))
        tensor = torch.load(open(tensorpath, 'rb'))
        return tensor.cpu().unsqueeze(dim=0).numpy()

    def run_inference(self, k=5):
        self.embed_query()
        embedded_query = self.find_tensors_and_convert()
        dist, neighbors = self.search_full_index(embedded_query, k)
        jsonf = self.find_jsonfile()
        knn_imagepaths = [Path(jsonf[str(neighbor)]) for neighbor in neighbors[0]]

        os.makedirs(self.outputpath, exist_ok=True)
        for path in knn_imagepaths:
            shutil.copyfile(path, os.path.join(self.outputpath, path.name))

    def run_filtered_inference(self, exhaust=10, k=1000):
        self.embed_query()
        embedded_query = self.find_tensors_and_convert()
        jsonf = self.find_jsonfile()
        dist, neighbors = self.search_full_index(embedded_query, k)
        last_distance = 9999
        knn_imagepaths = []
        seen_distances = []
        for index, neighbor in enumerate(neighbors[0]):
            if dist[0][index] not in seen_distances:
                knn_imagepaths.append(jsonf[str(neighbors[0][index])])
                seen_distances.append(dist[0][index])
                if len(knn_imagepaths)>exhaust:
                    break
        os.makedirs(self.outputpath, exist_ok=True)
        for path in knn_imagepaths:
            path = Path(path)
            shutil.copyfile(path, os.path.join(self.outputpath, path.name))


class KNNIndexTrainer:
    def __init__(self, inputpath, outputpath, batchsize):
        self.inputpath = Path(inputpath)
        self.outputpath = Path(outputpath)
        self.writepath = os.path.join(self.inputpath.parent, "index")
        self.vectors = self.find_tensors_and_convert()
        self.index = faiss.index_factory(self.vectors.shape[1], "IVF500,Flat")
        self.batchsize = batchsize

    def find_tensors_and_convert(self):
        glob = Path(self.inputpath).rglob("**/tensors.pt")
        glob = sorted(glob, key=os.path.getctime)
        tensors = [torch.load(open(path, 'rb')) for path in glob]
        for index, tensor in enumerate(tensors):
            if len(tensor.shape) < 2:
                tensors[index] = tensor.unsqueeze(dim=0)

        return torch.cat(tensors).cpu().numpy()

    def find_paths_and_merge(self):
        """I need to keep track of the index and respective filepaths."""
        glob = Path(self.inputpath).rglob("**/paths.json")
        glob = sorted(glob, key=os.path.getctime)
        jsonf = {}
        index = 0
        for p in glob:
            jsondata = json.load(open(p, 'r'))
            for k, path in jsondata.items():
                jsonf[index] = path
                index += 1
        with open(os.path.join(self.writepath, "filepath_index.json"), 'w') as outfile:
            json.dump(jsonf, outfile)

    def train_index(self):
        self.index.train(self.vectors[0:self.batchsize])

    def write_index(self, filename="trained.index"):
        os.makedirs(self.writepath, exist_ok=True)
        faiss.write_index(self.index, os.path.join(self.writepath, filename))

    def read_index(self, filename="trained.index"):
        self.index = faiss.read_index(os.path.join(self.writepath, filename))

    def build_index(self):
        self.train_index()
        self.write_index()
        n_batches = self.vectors.shape[0] // self.batchsize
        for i in range(n_batches):
            self.read_index()
            self.index.add_with_ids(self.vectors[i * self.batchsize:(i + 1) * self.batchsize],
                                    np.arange(i * self.batchsize, (i + 1) * self.batchsize))
            self.write_index(f"block_{i}.index")
        self.read_index()
        block_fnames = [os.path.join(self.writepath, f"block_{b}.index") for b in range(n_batches)]
        merge_ondisk(self.index, block_fnames, "merged_index.ivfdata")
        self.write_index("populated.index")
        self.find_paths_and_merge()

    def search_full_index(self, vectors, k):
        self.read_index("populated.index")
        self.index.nprobe = 80
        distances, neighbors = self.index.search(vectors, k)
        return distances, neighbors


def main():
    """The main function can be called from the command line to build an index."""
    train = True
    train = False  # transform to cmd-line-option later
    if train:
        kwargs = parse_args()
        knn = KNNIndexTrainer(**kwargs)
        knn.build_index()
    else:
        # knn = KNNIndexInference(inputpath="/home/frank/data/celador/query3", dataset="/home/frank/data/celador")
        knn = KNNIndexInference(inputpath="/home/frank/data/.60k/query2", dataset="/home/frank/data/.60k")
        knn.run_filtered_inference()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
