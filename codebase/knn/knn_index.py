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

import codebase.data.hashset
from codebase.knn.embedder import Embedder


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description="Embed a folder of images.")
    parser.add_argument("-i", "--inputpath", type=str, help="folder to process")
    parser.add_argument("-o", "--outputpath", type=str, help="folder for output")
    parser.add_argument(
        "-bs", "--batchsize", type=int, help="batchsize for embedder model"
    )

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
            return faiss.read_index(indexfile.as_posix(), faiss.IO_FLAG_ONDISK_SAME_DIR)
        except StopIteration:
            raise FileNotFoundError("No index file found. Exiting.")

    def find_jsonfile(self):
        try:
            filep = next(Path(self.dataset).rglob("**/filepath_index.json"))
            return json.load(open(filep, "r"))
        except StopIteration:
            raise FileNotFoundError("No json file found. Exiting.")

    def embed_query(self):
        embedder = Embedder(
            inputpath=self.inputpath,
            outputpath=self.outputpath,
            batchsize=self.batchsize,
        )
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
        tensor = torch.load(open(tensorpath, "rb"))
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

        net = SentenceTransformer("all-MiniLM-L6-v2")
        net.max_seq_length = (
            30  # quadratic increase of transformer nodes with increasing input size!
        )
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


class KNNHashIndexTrainer:
    def __init__(self, hashset: codebase.data.hashset.HashSet, batchsize: int = 30000):
        self.hashset = hashset
        self.batchsize = batchsize
        self.outputpath = None

        self.embd_outputpath = Path(self.hashset._target_embd_knn_index_subdir)
        self.caption_outputpath = Path(self.hashset._target_cap_knn_index_subdir)
        self.combi_outputpath = Path(self.hashset._target_combi_knn_index_subdir)

        self.embd_vectors = None
        self.cap_vectors = None
        self.combi_vectors = None

        print("done merging the vectors")

        self.embd_index = faiss.index_factory(self.embd_vectors.shape[1], "IVF400,Flat")
        self.cap_index = faiss.index_factory(self.cap_vectors.shape[1], "IVF400,Flat")
        self.combi_index = faiss.index_factory(
            self.combi_vectors.shape[1], "IVF400,Flat"
        )

        self.mode_triplets = {
            "embd": {
                "out": self.embd_outputpath,
                "vectors": self.embd_vectors,
                "index": self.embd_index,
            },
            "caption": {
                "out": self.caption_outputpath,
                "vectors": self.cap_vectors,
                "index": self.cap_index,
            },
            "combi": {
                "out": self.combi_outputpath,
                "vectors": self.combi_vectors,
                "index": self.combi_index,
            },
        }

    def train_index(self, mode: str):
        # TODO add the below path to the triplet, load here, dump the hashset in init. make sure each index is del before starting the next
        self.mode_triplets[mode]["vectors"] = torch.load(
            os.path.join(
                self.hashset._target_embd_knn_tensor_subdir, "embd_vectors_combined.pt"
            )
        )
        self.mode_triplets[mode]["index"].train(
            self.mode_triplets[mode]["vectors"][0 : self.batchsize]
        )

    def write_index(self, mode: str, filename="trained.index"):
        os.makedirs(self.mode_triplets[mode]["out"], exist_ok=True)
        faiss.write_index(
            self.mode_triplets[mode]["index"],
            os.path.join(self.mode_triplets[mode]["out"], filename),
        )

    def read_index(self, mode: str, filename="trained.index"):
        self.mode_triplets[mode]["index"] = faiss.read_index(
            os.path.join(self.mode_triplets[mode]["out"], filename)
        )

    def build_indexes(self):
        for mode in self.mode_triplets.keys():
            self.train_index(mode=mode)
            self.write_index(mode=mode)
            n_batches = self.mode_triplets[mode]["vectors"].shape[0] // self.batchsize
            for i in range(n_batches):
                self.read_index(mode=mode)
                self.mode_triplets[mode]["index"].add_with_ids(
                    self.mode_triplets[mode]["vectors"][
                        i * self.batchsize : (i + 1) * self.batchsize
                    ],
                    np.arange(i * self.batchsize, (i + 1) * self.batchsize),
                )
                self.write_index(mode=mode, filename=f"block_{i}.index")
            self.read_index(mode=mode)
            block_fnames = [
                os.path.join(self.mode_triplets[mode]["out"], f"block_{b}.index")
                for b in range(n_batches)
            ]
            merge_ondisk(
                self.mode_triplets[mode]["index"],
                block_fnames,
                os.path.join(self.mode_triplets[mode]["out"], "merged_index.ivfdata"),
            )
            self.write_index(mode=mode, filename="populated.index")
            for block in block_fnames:
                os.remove(block)


class KNNIndexTrainer:
    def __init__(
        self,
        inputpath,
        batchsize,
        outputpath=None,
    ):
        self.inputpath = Path(inputpath)
        if outputpath:
            self.outputpath = Path(outputpath)
        else:
            self.outputpath = self.inputpath
        self.writepath = os.path.join(self.inputpath.parent, "index")
        self.vectors = self.find_tensors_and_convert()
        self.index = faiss.index_factory(self.vectors.shape[1], "IVF400,Flat")
        self.batchsize = batchsize

    def find_tensors_and_convert(self):
        glob = Path(self.inputpath).rglob("**/tensors.pt")
        glob = sorted(glob, key=os.path.getctime)[:500000]
        tensors = [torch.load(open(path, "rb")).cpu() for path in glob]
        tensors = tensors  # [:int(0.5 * len(tensors))]
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
            jsondata = json.load(open(p, "r"))
            for k, path in jsondata.items():
                jsonf[index] = path
                index += 1
        with open(os.path.join(self.writepath, "filepath_index.json"), "w") as outfile:
            json.dump(jsonf, outfile)

    def train_index(self):
        self.index.train(self.vectors[0 : self.batchsize])

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
            self.index.add_with_ids(
                self.vectors[i * self.batchsize : (i + 1) * self.batchsize],
                np.arange(i * self.batchsize, (i + 1) * self.batchsize),
            )
            self.write_index(f"block_{i}.index")
        self.read_index()
        block_fnames = [
            os.path.join(self.writepath, f"block_{b}.index") for b in range(n_batches)
        ]
        merge_ondisk(
            self.index,
            block_fnames,
            os.path.join(self.writepath, "merged_index.ivfdata"),
        )
        self.write_index("populated.index")
        for block in block_fnames:
            os.remove(block)

        self.find_paths_and_merge()


def main():
    """The main function can be called from the command line to build an index."""
    """    train = True
    # train = False  # transform to cmd-line-option later
    if train:
        kwargs = parse_args()
        knn = KNNIndexTrainer(**kwargs)
        knn.build_index()
    else:
        knn = KNNIndexInference(inputpath="/home/frank/data/celador/knn/knn_inference/query_4",
                                dataset="/home/frank/data/celador")
        knn.run_filtered_inference()
    torch.cuda.empty_cache()"""
    hashset = codebase.data.hashset.HashSet(
        "/home/frank/ssd/backup/datasets/hash/dataset"
    )
    hashindex = KNNHashIndexTrainer(hashset)
    # hashindex.build_indexes()


if __name__ == "__main__":
    main()
