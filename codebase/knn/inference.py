# https://davidefiocco.github.io/nearest-neighbor-search-with-faiss/
import argparse
from pathlib import Path

import faiss
import torch

import codebase
from codebase.data.hashset import HashSet
from codebase.data.sample import Sample



def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description='Embed a folder of images.')
    parser.add_argument('-i', '--inputpath', type=str, help='folder to process')
    parser.add_argument('-o', '--outputpath', type=str, help='folder for output')
    parser.add_argument('-bs', '--batchsize', type=int, help='batchsize for embedder model')

    args = parser.parse_args()
    kwargs = vars(args)  # convert namespace to dictionary
    return kwargs


class HashInferenceResult:
    def __init__(self,
                 query: codebase.data.sample.Sample,
                 searchmode: str = "both",
                 result_indices=None,
                 result_distances=None,
                 result_samples=None):

        if result_samples is None:
            result_samples = []
        if result_distances is None:
            result_distances = []
        if result_indices is None:
            result_indices = []
        self.query = query
        self.searchmode = searchmode
        self.result_indices = result_indices
        self.result_distances = result_distances
        self.result_samples = result_samples


class KNNHashIndexInference:
    """Hash index inference class. Runs from already processed files, using Sample class."""

    def __init__(self,
                 searchmode: str = "both",
                 hashset=None,
                 ):

        self.searchmode = searchmode
        self.hashset = hashset
        self.query_temp = {}  # store temporarily
        self.index = self.find_indexfile(self.hashset, self.searchmode)  # decides in inference

    def gather_query_tensors(self, samplelist: list[Sample]):
        """get tensors as list (for later on concat) and store reference hash to index in tensor"""
        query_tensors = []
        for k, sample in enumerate(samplelist):
            query_tensor = sample.get_embedding(embeddingtype=self.searchmode)
            query_tensors.append(query_tensor)
            self.query_temp[sample.id] = k

        return query_tensors

    def inference(self, samplelist: list[Sample], hashset, searchmode=None, exhaust=20):
        """searchmode: both | embd | caption
        query: list of hash-ids (str)"""
        if searchmode:
            self.index = self.find_indexfile(hashset=hashset, searchmode=searchmode)

        self.query_temp = {}  # reset

        tensors = self.gather_query_tensors(samplelist=samplelist)
        tensors = torch.cat(tensors, dim=0).cpu().numpy()

        neighbor_indices, neighbor_dists = self.exhaustive_search(tensors, exhaust=exhaust)

        results = []
        for row, query in enumerate(samplelist):
            results.append(HashInferenceResult(query=query,
                                               searchmode=self.searchmode,
                                               result_indices=neighbor_indices[row],
                                               result_distances=neighbor_dists[row]))

        return results

    def exhaustive_search(self, tensor_embd, exhaust=20, k=100):
        """This returns n=exhaust many results and their respective distances as lists.
        IMPORTANT: returns results for batches of images!
        Args:
            tensor_embd:    embedding tensor, stacked along dim-0
            exhaust:        cut-off after this many unique neighbors
            k:              return this many neighbors from knn-index"""

        dists, neighbors = self.search_full_index(tensor_embd, k)
        knn_imagepaths = []
        seen_distances = []

        for query_idx, results in enumerate(neighbors):
            result_indices = []
            result_distances = []
            for index, neighbor in enumerate(results):
                if dists[query_idx][index] not in result_distances:
                    result_indices.append(neighbor)
                    result_distances.append(dists[query_idx][index])
                    if len(result_indices) > exhaust:
                        break
            knn_imagepaths.append(result_indices)
            seen_distances.append(result_distances)
        return knn_imagepaths, seen_distances

    def find_indexfile(self, hashset, searchmode=None):
        """finds an index file for which_index = caption | embd | both"""
        if searchmode:
            self.searchmode = searchmode
        index_subdir = ""
        if self.searchmode == "caption":
            index_subdir = hashset._target_cap_knn_index_subdir
        if self.searchmode == "embd":
            index_subdir = hashset._target_embd_knn_index_subdir
        if self.searchmode == "both":
            index_subdir = hashset._target_combi_knn_index_subdir
        try:
            indexfile = next(Path(index_subdir).rglob("populated.index"))
            return faiss.read_index(indexfile.as_posix(), faiss.IO_FLAG_ONDISK_SAME_DIR)
        except StopIteration:
            raise FileNotFoundError(f"No index file found for {self.searchmode}. Exiting.")

    def search_full_index(self, vectors, k):
        self.index.nprobe = 80
        distances, neighbors = self.index.search(vectors, k)
        return distances, neighbors


def main():
    """The main function can be called from the command line to build an index."""

    hashset = HashSet("/home/frank/ssd/backup/datasets/hash/dataset")

    querylist = ["0ee2bc44facc16a9e436479e58036929", "1da5525b8028c4d1f36fb465785e9a2f"]
    samplelist = hashset.prepare_samplelist_from_query(querylist)

    """queryset = HashSet("/home/frank/ssd/backup/datasets/atf/dataset")
    querylist = ["2f37682102e4b31da05c223bc1c3f2ef", "6e92be19905b1fadf5186387d4735e4e",
                 "63ec56748cc6ca1440f778757dde3a2b"]
    samplelist = queryset.prepare_samplelist_from_query(querylist)"""

    """hashset = HashSet("/home/frank/datasets/celador/dataset")
    querylist = ["03ace42e4655293cf1e4c4ccfc120a22", "0a1772298d4d03e166f76d1d74d6221a"]
    samplelist = hashset.prepare_samplelist_from_query(querylist)"""

    searchindex = KNNHashIndexInference(searchmode="both", hashset=hashset)
    resultlist = searchindex.inference(hashset=hashset, samplelist=samplelist, exhaust=9)

    # imported here to avoid circular import error
    from codebase.data.visualize import SampleVisualizer

    vis = SampleVisualizer()
    vis = SampleVisualizer("/home/frank/data/apps/.hash_plot_output")

    for res in resultlist:
        vis.save_knn_result_grid(hashset=hashset, result=res)


if __name__ == '__main__':
    main()
