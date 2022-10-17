import os
from pathlib import Path
from typing import cast, Callable

from PIL import Image
import PIL
from torch.utils.data import Dataset
from codebase.data.transforms import efficientnet_transforms
from torchvision import datasets
from torchvision.datasets.folder import has_file_allowed_extension


def find_classless(directory: str):
    """returns the name of the directory as class for IFSet class"""
    classes = [Path(directory).name]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
    """uses parent folder for finding images for IFSet class."""
    directory = os.path.expanduser(directory)
    # there's a lot deleted here. see folder.py in datasets.ImageFolder/FolderDataset
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]
    is_valid_file = cast(Callable[[str], bool], is_valid_file)
    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = directory
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
                    if target_class not in available_classes:
                        available_classes.add(target_class)
    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)
    return instances


class IFSet(datasets.ImageFolder):
    """Functions overwrite the originals from torchvision.datasets.folder
    Embedder has to handle each subdirectory as own entity: crashes if called on parent folder - json files too long
    problem: ImageFolder set uses subdirs as classnames
    solution: pass name of subdir as classname. overwrite functions"""

    def __getitem__(self, index: int):
        """returns image path instead of class info. class info irrelevant"""
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path  # need path instead of return sample, target (class irrelevant for now)

    def find_classes(self, directory: str):
        return find_classless(directory)

    @staticmethod
    def make_dataset(directory, class_to_idx, extensions=None, is_valid_file=None):
        """adjusted method from datasets.folder.py. Overrides looking for classes in subdirs to allow
        sequential processing of multiple folders."""
        if class_to_idx is None:
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def sanity_check_imagefiles(self):
        for sample in self.samples:
            path, cla = sample
            try:
                im = Image.open(path)
                im.verify()
                im.close()
            except:
                # print(f"Broken image file {path}. Removing.")
                os.remove(path)
                self.samples.remove(sample)


class SBERTSet(Dataset):
    """A dataset that loads the JSON-files with captions and turns them into tensors for later-on knn indexing."""
    def __init__(self, inputpath):
        self.inputpath = Path(inputpath)
        self.jsonf = self.load_jsonfile()
        self.paths = list(self.jsonf.keys())
        self.captions = list(self.jsonf.values())

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, item):
        """returns caption and path"""
        return self.captions[item], self.paths[item]

    def load_jsonfile(self):
        """loads json file from folder. convention: jsonfile is key (index) - value (path)"""
        import json
        try:
            filep = next(Path(self.inputpath).rglob("*.json"))
            return json.load(open(filep, 'r'))
        except StopIteration:
            raise FileNotFoundError("No json file found. Exiting.")