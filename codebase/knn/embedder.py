import argparse
import os
import torchvision
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import warnings
from codebase.data.transforms import efficientnet_transforms
import json
from pathlib import Path

from codebase.data.datasets import IFSet, SBERTSet

warnings.filterwarnings('ignore')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.cuda.empty_cache()


# print(f'Using {device} for inference')


def parse_args():
    """Echo the input arguments to standard output"""
    parser = argparse.ArgumentParser(description='Embed a folder of images.')
    parser.add_argument('-i', '--inputpath', type=str, help='folder to process')
    parser.add_argument('-o', '--outputpath', type=str, help='folder for output')
    parser.add_argument('-bs', '--batchsize', type=int, help='batchsize for embedder model')

    args = parser.parse_args()
    kwargs = vars(args)  # convert namespace to dictionary
    return kwargs


class Embedder:
    """A class to embed images in bulk. Built on EfficientNetB0.
    Args:
        inputpath:          inputpath
        outputpath:         where to store
        batchsize:          batchsize
        """

    def __init__(self, inputpath=".", outputpath=".", batchsize=300):
        """lazy init. enables one-by-one processing of subdirectories via update_params() once checks have passed.
        pro: only setup EfficientNet model once."""
        self.net = None
        self.inputpath = Path(inputpath)
        self.outputpath = Path(outputpath)
        self.batchsize = batchsize
        self.dataloader = None
        self.filepaths = {}
        self.all_paths = []
        self.tensorpaths = {}
        self.write_path = os.path.join(self.outputpath, self.inputpath.name)

    def update_params(self, inputpath, outputpath, batchsize):
        self.inputpath = Path(inputpath)
        self.outputpath = Path(outputpath)
        self.batchsize = batchsize
        self.dataloader = self.dataloader_setup()
        self.filepaths = {}
        self.all_paths = []
        self.tensorpaths = {}
        self.write_path = os.path.join(self.outputpath, self.inputpath.name)

    def init_model(self):
        efficientnet = torchvision.models.efficientnet_b0(
            weights=torchvision.models.efficientnet.EfficientNet_B0_Weights.DEFAULT)
        self.net = torch.nn.Sequential(*list(efficientnet.children())[:-1])

    def dataloader_setup(self):
        dataset = IFSet(self.inputpath.as_posix(), transform=efficientnet_transforms)
        # dataset.sanity_check_imagefiles()  # avoid PIL.UnidentifiedImageError
        return DataLoader(dataset=dataset, batch_size=self.batchsize, num_workers=16)

    def update_filepaths(self, index, paths):
        self.tensorpaths[str(index)] = os.path.join(self.write_path, f"batch_{index}.pt")
        self.filepaths[f"batch_{index}"] = paths
        for path in paths:
            self.all_paths.append(path)

    def merge_filepaths(self):
        merged_filepaths = {}
        for index, value in enumerate(self.all_paths):
            merged_filepaths[str(index)] = value
        self.filepaths = merged_filepaths

    def write_filepaths(self, filepath=None):
        if filepath:
            with open(filepath, 'w') as fp:
                json.dump(self.filepaths, fp)
        else:
            with open(os.path.join(self.write_path, f"{self.inputpath.name}.json"), 'w') as fp:
                json.dump(self.filepaths, fp)

    def merge_tensors(self):
        glob = Path(self.write_path).rglob("batch_*.pt")
        glob = sorted(glob, key=os.path.getctime)

        tensors = [torch.load(open(path, 'rb')) for path in glob]
        for index, tensor in enumerate(tensors):
            if len(tensor.shape) < 2:  # tensors must have the same size, but single-entry tensors (one image) have d=1
                tensors[index] = tensor.unsqueeze(dim=0)
        merge = torch.cat(tensors)
        torch.save(merge, os.path.join(self.write_path, f"tensors.pt"))

        for pt in glob:
            os.remove(pt)

    def process_output(self):
        self.merge_tensors()
        self.merge_filepaths()
        self.write_filepaths(os.path.join(self.write_path, "paths.json"))

    def embed(self):
        self.net.eval().to(device)
        os.makedirs(self.write_path, exist_ok=True)
        # print(f"Embedding {len(self.dataloader) * self.batchsize} images in {len(self.dataloader)} batches.")
        try:
            for index, batch in enumerate(self.dataloader):
                tensors, paths = batch
                tensors = tensors.to(device)
                with torch.no_grad():
                    outputs = self.net(tensors)
                    outputs = outputs.squeeze()
                    torch.save(outputs, os.path.join(self.write_path, f"batch_{index}.pt"))
                self.update_filepaths(index, paths)
        except:
            print(f"error in: {self.inputpath.name}")


class SentenceEmbedder(Embedder):

    def init_model(self):
        from sentence_transformers import SentenceTransformer
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.net = SentenceTransformer('all-MiniLM-L6-v2')
        self.net.max_seq_length = 30  # quadratic increase of transformer nodes with increasing input size!

    def embed(self):
        os.makedirs(self.write_path, exist_ok=True)
        try:
            for index, batch in enumerate(self.dataloader):
                captions, paths = batch
                # captions = captions.to(device)
                outputs = self.net.encode(captions, convert_to_tensor=True)
                # outputs = outputs.squeeze()
                # print(f"shape of the output is:{outputs.shape}")
                torch.save(outputs, os.path.join(self.write_path, f"batch_{index}.pt"))
                self.update_filepaths(index, paths)
        except:
            print(f"error in: {self.inputpath.name}")

    def dataloader_setup(self):
        # init new dataloader for jsonfile-content
        dataset = SBERTSet(self.inputpath.as_posix())
        return DataLoader(dataset=dataset, batch_size=self.batchsize, num_workers=4)

    def update_params(self, inputpath, outputpath, batchsize):
        self.inputpath = Path(inputpath)
        self.outputpath = Path(inputpath)
        self.batchsize = batchsize
        self.dataloader = self.dataloader_setup()
        self.filepaths = {}
        self.all_paths = []
        self.tensorpaths = {}
        self.write_path = self.inputpath


def main():
    kwargs = parse_args()
    parentfolder = kwargs.pop("inputpath")
    subdirectories = [p for p in Path(parentfolder).iterdir() if p.is_dir()]
    assert len(subdirectories) > 0, "No subdirectories found! Provide a parent path or move images to subfolder."
    sbert = True
    if sbert:
        embedder = SentenceEmbedder()
    else:
        embedder = Embedder()
    embedder.init_model()
    for sub in tqdm(subdirectories):
        embedder.update_params(sub, **kwargs)
        if os.path.exists(embedder.write_path) and len(os.listdir(embedder.write_path)) == 2:
            print(f"Directory {embedder.inputpath.name} already processed. Moving on.")
            continue
        else:
            embedder.embed()
            embedder.process_output()
            pass
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
