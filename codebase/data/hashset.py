import json
import os
from pathlib import Path
from typing import cast, Callable

from PIL import Image
from pprint import pprint
import PIL
# from torch.utils.data import Dataset
# from codebase.data.transforms import efficientnet_transforms
# from torchvision import datasets
# from torchvision.datasets.folder import has_file_allowed_extension

class HashSet():
    """This class combines a dataset with img, img embd, cap, cap embd in one access structure."""

    def __init__(self,
        dataset_directory: str
    ):
        self.subdirectories = [p for p in Path(dataset_directory).iterdir() if p.is_dir()]
        self.caption_structure = self.find_captions()
        self.embd_structure = self.find_embds()
        
    def find_captions(self):
        """find paths to data structures containing caption embeddings"""
        caption_structure = {}
        for p in self.subdirectories:
            if 'caption' in p.as_posix():
                caption_structure["caption_directory"]=p
                caption_json_paths = sorted(Path(p).rglob("**/*captions.json"), key=os.path.getctime)
                caption_indices = sorted(Path(p).rglob("**/*paths.json"), key=os.path.getctime)
                caption_structure["caption_json_paths"]=caption_json_paths
                caption_structure["caption_indices"]=caption_indices
                return caption_structure

    def find_embds(self):
        """find paths to data structures containing image embeddings"""
        embd_structure = {}
        for p in self.subdirectories:
            if 'knn' in p.as_posix():
                embd_structure["embd_directory"]=p
                embd_paths = sorted(Path(p).rglob("**/*paths.json"), key=os.path.getctime)
                embd_structure["embd_json_paths"]=embd_paths
                return embd_structure


    def make_dataset(self):
        samples = {}
        for jsonf in self.caption_structure["caption_json_paths"]:
            jsondata = json.load(open(jsonf), 'r')
            for key, value in jsondata.items():
                sample = Sample()

hash = HashSet("/Users/FrankTheTank/data/celador")
hash.subdirectories
pprint(hash.caption_structure)

class Sample():
    def __init__(self,
        id, # potentially hash in the future
        img_path = None,
        embd_tensor_path = None,
        embd_tensor_index = None,
        caption = None,
        cap_tensor_path = None,
        cap_tensor_index = None,
        classification = None
    ):
        self.id = id
        self.img_path = img_path
        self.embd_tensor_path = embd_tensor_path
        self.embd_tensor_index = embd_tensor_index
        self.caption = caption
        self.cap_tensor_path = cap_tensor_path
        self.cap_tensor_index = cap_tensor_index
        self.classification = classification

    def serialize(self):
        return vars(self)


