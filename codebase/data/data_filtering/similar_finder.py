import argparse
import json
import os
import shutil
from pathlib import Path
import numpy as np
import torch

from codebase.knn.embedder import Embedder
from codebase.knn.knn_index import KNNIndexInference


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(
        description="Search similar images for a given json-file with filepaths."
    )
    parser.add_argument("-i", "--inputpath", type=str, help="json to process")
    parser.add_argument("-o", "--outputpath", type=str, help="folder for output")
    parser.add_argument(
        "-k", "--neighbors", type=str, default=100, help="how many neighbors to return"
    )
    parser.add_argument(
        "-idx", "--index", type=str, help="index folder to match against"
    )
    parser.add_argument(
        "-bs", "--batchsize", default=300, type=int, help="batchsize for embedder model"
    )

    args = parser.parse_args()
    kwargs = vars(args)  # convert namespace to dictionary
    return kwargs


class SimilarFinder:
    """A convenience class to pass a list of images, and get the nearest neighbors based on EfficientNet-Embeddings."""

    def __init__(self, inputpath, neighbors, index, outputpath=None, batchsize=100):
        self.inputpath = Path(inputpath)
        self.outputpath = Path(outputpath)
        self.batchsize = batchsize
        self.index = index  # path to dataset
        self.neighbors = neighbors
        self.query_dir = self.make_query_dir()
        self.result_dir = self.make_result_dir()

    def make_result_dir(self):
        resultdir = Path(os.path.join(self.query_dir, "results"))
        os.makedirs(resultdir, exist_ok=True)
        return resultdir

    def make_query_dir(self):
        dirs = [f for f in self.outputpath.iterdir() if f.is_dir()]
        dirs = sorted(dirs, key=os.path.getmtime)
        try:
            latest_index = dirs[-1].as_posix().split("_")[-1]
            new_index = str(int(latest_index) + 1)
            newdir = Path(dirs[-1].as_posix().replace(latest_index, new_index))
            os.makedirs(newdir, exist_ok=True)
        except IndexError:
            newdir = Path((os.path.join(self.outputpath, "query_0")))
            os.makedirs(newdir, exist_ok=True)
            return newdir
        return newdir

    def copy_over_query(self, imgpaths):
        for p in imgpaths:
            try:
                p = Path(p)
                shutil.copyfile(p, os.path.join(self.query_dir, p.name))
            except:
                print("oops")

    def copy_over_result(self, imgpaths, dists, cutoffdist=500):
        for k, p in enumerate(imgpaths):
            if dists[k] < cutoffdist:
                p = Path(p)
                shutil.copyfile(p, os.path.join(self.result_dir, p.name))

    def find_matches(self):
        """Args: paths to images
        Returns: matplotlib representation of search results"""

        imgps = self.load_jsonfile()
        imgps = list(imgps.values())
        self.copy_over_query(imgps)
        knn = KNNIndexInference(inputpath=self.query_dir, dataset=self.index)
        paths, dist = knn.inference_for_dataset_cleanup()
        return paths, dist

    def load_jsonfile(self):
        """loads json file from folder. convention: jsonfile is key (index) - value (path)"""
        try:
            filep = next(Path(self.inputpath).rglob("*.json"))
            return json.load(open(filep, "r"))
        except StopIteration:
            raise FileNotFoundError("No json file found. Exiting.")


def main():
    """The main function can be called from the command line."""
    kwargs = parse_args()
    simfinder = SimilarFinder(**kwargs)
    neighbors, dists = simfinder.find_matches()
    for k, neighbor in enumerate(neighbors):
        simfinder.copy_over_result(neighbor, dists[k])

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
