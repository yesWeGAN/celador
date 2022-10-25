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


def find_jsonfile_index(folder, imgpath):
    """used to retrieve the index for a given path in json files mapping path-to-tensorindex"""
    try:
        filep = next(Path(folder).rglob("paths.json"))
        jsonf = json.load(open(filep, 'r'))
        # map path back to index
        return list(jsonf.keys())[list(jsonf.values()).index(imgpath)]
    except StopIteration:
        raise FileNotFoundError("No json file found. Exiting.")


def load_tensorfile_index(path, index):
    """used to load the pre-processed tensor given an index in the tensor file"""
    try:
        filep = next(Path(path).rglob("*tensors.pt"))
        tensor = torch.load(open(filep, 'rb'))
        return tensor.cpu().numpy()[int(index)]
    except StopIteration:
        raise FileNotFoundError("No json file found. Exiting.")


def register_datasets():
    """Register a dataset with the 5-tuple: ("images", img_index, img_embd, cap_index, cap_embd)"""
    datasets = {
        "celador": {
            "images": "/home/frank/data/celador/img/bing",
            "img_index": "/home/frank/data/celador/knn/index",
            "img_embd": "/home/frank/data/celador/knn/embd",
            "cap_index": "/home/frank/data/celador/caption/index",
            "cap_embd": "/home/frank/data/celador/caption/captions",
            "stout": "/home/frank/data/celador/inference"
        },
        "60k": {
            "images": "/home/frank/data/.60k/img",
            "img_index": "/home/frank/data/.60k/knn/index",
            "img_embd": "/home/frank/data/.60k/knn/embd",
            "cap_index": "/home/frank/data/.60k/caption/index",
            "cap_embd": "/home/frank/data/.60k/caption/captions",
            "stout": "/home/frank/data/.60k/inference"
        },
        "1m": {
            "images": "/home/frank/ssd/backup/data/instagram/bogan",
            "img_index": "/home/frank/ssd/backup/data/instagram/knn/index",
            "img_embd": "/home/frank/ssd/backup/data/instagram/knn/embd",
            "cap_index": "/home/frank/ssd/backup/data/instagram/index",
            "cap_embd": "/home/frank/ssd/backup/data/instagram/caption",
            "stout": "/home/frank/ssd/backup/data/instagram/inference"
        }}
    return datasets


class KNNIndexInference:
    """Reshape this class to be truly useful."""

    def __init__(self, batchsize=100):

        self.batchsize = batchsize
        self.searchmode = None
        self.dataset = None  # is a dict from registered datasets
        self.registered_datasets = register_datasets()
        self.mode = None

    def infer_mode(self, query: list):
        if os.path.isfile(query[0]):
            for q in query:
                if os.path.isfile(q):
                    continue
                else:
                    raise FileNotFoundError(f"One or more image files not found: {q}")
            self.mode = "from_images"
        elif query[0] is isinstance(str):
            self.mode = "from_caption"
        else:
            raise TypeError("Query does not have ")

    def infer_dataset(self, query: list):
        for key, value in self.registered_datasets.items():
            if value['images'] in query[0]:
                for q in query:
                    if value['images'] in q:
                        continue
                    else:
                        raise FileNotFoundError(f"One or more image files is not from infered dataset {key}: {q}")
                return self.registered_datasets[key]

    def gather_tensors(self, query, searchmode, cat = True):
        """This method gathers tensors for both caption and img from pre-processed.
        Args:
            query:   list of image paths in "from_images" mode
            cat:    True | False to concatenate (do not for later on cosine similarity processing)
        Returns:
            img_tensors, cap_tensors    (either empty or stacked tensors along dim-0, depending on self.searchmode)"""

        img_tensors = []
        cap_tensors = []

        for q in query:
            q = Path(q)
            parent = q.parent

            if searchmode in ["both", "image"]:
                img_tensor_dir = os.path.join(self.dataset["img_embd"], parent.name)
                img_json_index = find_jsonfile_index(img_tensor_dir, q.as_posix())
                img_tensor = load_tensorfile_index(img_tensor_dir, img_json_index)
                img_tensors.append(img_tensor)

            if searchmode in ["both", "caption"]:
                cap_tensor_dir = os.path.join(self.dataset["cap_embd"], parent.name)
                cap_json_index = find_jsonfile_index(cap_tensor_dir, q.as_posix())
                cap_tensor = load_tensorfile_index(cap_tensor_dir, cap_json_index)
                cap_tensors.append(cap_tensor)

        if cat:
            if searchmode in ["both", "image"]:
                img_tensors = torch.cat(img_tensors)
            if searchmode in ["both", "caption"]:
                cap_tensors = torch.cat(cap_tensors)

        return img_tensors, cap_tensors

    def inference(self, query: list, searchmode=None, output=None):
        """searchmode: both | image | caption"""
        # first, determine type of input
        self.infer_mode(query=query)

        # now, infer datset
        if self.mode == "from_images":
            self.dataset = self.infer_dataset(query=query)

        # define search mode: image, caption, or both (default)
        if searchmode:
            self.searchmode = searchmode
        else:
            self.searchmode = "both"

        # now, gather tensors. either from existing, or embed fresh (MISSING TO DO)
        img_tensors, cap_tensors = self.gather_tensors(query, self.searchmode)

        img_neighbors = []
        cap_neighbors = []

        if self.searchmode in ["both", "image"]:
            img_neighbors, _ = self.exhaustive_search(img_tensors, which_index="img_index")

        if self.searchmode in ["both", "caption"]:
            cap_neighbors, _ = self.exhaustive_search(img_tensors, which_index="cap_index")

        # this here only happens because we do not yet have a common index
        if self.searchmode == "both":
            intersection = list(set(img_neighbors).intersection(cap_neighbors))
            img_tensors = self.gather_tensors(intersection, "image", cat=False)
            cap_tensors = self.gather_tensors(intersection, "caption", cat=False)
            print(f"Found intersection in img / cap in {len(intersection)} files:")
            for i in intersection:
                print(i)
            # find_cosines(img_neighbors, cap_neighbors)

    def exhaustive_search(self, tensor_embd, which_index, exhaust=100, k=5000):
        """This returns n=exhaust many results and their respective distances as lists.
        IMPORTANT: returns results for batches of images!
        Args:
            tensor_embd:    embedding tensor, stacked along dim-0
            which_index:    caption or img
            exhaust:        cut-off after this many unique neighbors
            k:              return this many neighbors from knn-index"""

        jsonf = self.find_jsonfile(which_index)
        self.index = self.find_indexfile(which_index)
        dists, neighbors = self.search_full_index(tensor_embd, k)
        knn_imagepaths = []
        seen_distances = []
        for idx, sample in enumerate(neighbors):
            sample_imagepaths = []
            sample_distances = []
            for index, neighbor in enumerate(sample):
                if dists[idx][index] not in sample_distances:
                    sample_imagepaths.append(jsonf[str(sample[index])])
                    sample_distances.append(dists[idx][index])
                    if len(sample_imagepaths) > exhaust:
                        break
            knn_imagepaths.append(sample_imagepaths)
            seen_distances.append(sample_distances)
        return knn_imagepaths, seen_distances

    def find_indexfile(self, which_index):
        """finds an index file for which_index = caption | img"""
        try:
            indexfile = next(Path(self.dataset[which_index]).rglob("populated.index"))
            return faiss.read_index(indexfile.as_posix(), faiss.IO_FLAG_ONDISK_SAME_DIR)
        except StopIteration:
            raise FileNotFoundError(f"No index file found for {which_index}. Exiting.")

    def find_jsonfile(self, which_index):
        """Retrieves the json-file for which_index = caption | img, mapping from knn-index to path"""
        try:
            filep = next(Path(self.dataset[which_index]).rglob("**/filepath_index.json"))
            return json.load(open(filep, 'r'))
        except StopIteration:
            raise FileNotFoundError(f"No json file found for {which_index}. Exiting.")

    def embed_query(self):
        embedder = Embedder(inputpath=self.inputpath, outputpath=self.outputpath, batchsize=self.batchsize)
        embedder.init_model()
        embedder.dataloader = embedder.dataloader_setup()
        embedder.embed()
        embedder.process_output()  # by the end of this, there will be a tensors.pt file in self.inputpath

    def search_full_index(self, vectors, k):
        self.index.nprobe = 80
        distances, neighbors = self.index.search(vectors, k)
        return distances, neighbors

    def find_tensors_and_convert(self):
        tensorpath = next(Path(self.inputpath).rglob("**/tensors.pt"))
        tensor = torch.load(open(tensorpath, 'rb'))
        return tensor.cpu().numpy()

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
        """This returns n=exhaust many results and their respective distances as lists"""
        self.embed_query()
        embedded_query = self.find_tensors_and_convert()
        jsonf = self.find_jsonfile()
        dist, neighbors = self.search_full_index(embedded_query, k)
        knn_imagepaths = []
        seen_distances = []
        for index, neighbor in enumerate(neighbors[0]):
            if dist[0][index] not in seen_distances:
                knn_imagepaths.append(jsonf[str(neighbors[0][index])])
                seen_distances.append(dist[0][index])
                if len(knn_imagepaths) > exhaust:
                    break
        return knn_imagepaths, seen_distances

    def inference_for_dataset_cleanup(self, exhaust=100, k=5000):
        """This returns n=exhaust many results and their respective distances as lists.
        IMPORTANT: returns results for batches of images!"""
        self.embed_query()
        embedded_query = self.find_tensors_and_convert()
        jsonf = self.find_jsonfile()
        dists, neighbors = self.search_full_index(embedded_query, k)
        knn_imagepaths = []
        seen_distances = []
        for idx, sample in enumerate(neighbors):
            sample_imagepaths = []
            sample_distances = []
            for index, neighbor in enumerate(sample):
                if dists[idx][index] not in sample_distances:
                    sample_imagepaths.append(jsonf[str(sample[index])])
                    sample_distances.append(dists[idx][index])
                    if len(sample_imagepaths) > exhaust:
                        break
            knn_imagepaths.append(sample_imagepaths)
            seen_distances.append(sample_distances)
        return knn_imagepaths, seen_distances

    def copy_results(self, knn_neighborpathlist, outputpath=None):
        """This copies search results into a given folder (defaults to self.outputpath.)"""
        if outputpath:
            writepath = Path(outputpath)
        else:
            writepath = self.outputpath
        os.makedirs(writepath, exist_ok=True)
        for path in knn_neighborpathlist:
            path = Path(path)
            shutil.copyfile(path, os.path.join(writepath, path.name))


class CaptionKNNIndexInference(KNNIndexInference):
    """Searches for nearest-neighbor embeddings on captions."""

    def __init__(self, querylist: list, dataset, inputpath="."):
        super().__init__(inputpath, dataset)
        self.captions = querylist

    def embed_query(self):
        from sentence_transformers import SentenceTransformer
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        net = SentenceTransformer('all-MiniLM-L6-v2')
        net.max_seq_length = 30  # quadratic increase of transformer nodes with increasing input size!
        embedded_query = net.encode(self.captions, convert_to_tensor=True)
        return embedded_query.cpu().numpy()

    def run_filtered_inference(self, exhaust=10, k=1000):
        """This returns n=exhaust many results and their respective distances as lists"""
        embedded_query = self.embed_query()
        jsonf = self.find_jsonfile()
        dist, neighbors = self.search_full_index(embedded_query, k)
        knn_imagepaths = []
        seen_distances = []
        for index, neighbor in enumerate(neighbors[0]):
            if dist[0][index] not in seen_distances:
                knn_imagepaths.append(jsonf[str(neighbors[0][index])])
                seen_distances.append(dist[0][index])
                if len(knn_imagepaths) > exhaust:
                    break
        return knn_imagepaths, seen_distances



def main():
    """The main function can be called from the command line to build an index."""
    inf = KNNIndexInference()
    q = ["/home/frank/data/celador/img/bing/american_tanks/3b43a774-6de3-4488-a45a-515ff066736c.jpg",
         "/home/frank/data/celador/img/bing/marines/0b4349c330d4d4347afb2c03ca68c437.jpg"]
    inf.inference(q)
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

