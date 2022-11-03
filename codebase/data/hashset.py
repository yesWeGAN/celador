import _hashlib
import json
import os
import shutil
from pathlib import Path
import hashlib
from PIL import Image
import PIL
from sample import Sample

from dataset_structures import celador_test_input_structure, generic_dataset_structure


class HashSetTransformer:
    """This class combines a dataset with img, img embd, cap, cap embd in one access structure.
    Used to transform existing datasets into a common format."""

    def __init__(self,
                 source_structure: dict
                 ):
        # set up the source dirs
        self.source_structure = source_structure
        self.sourcepath = Path(source_structure["sourcepath"])
        self._source_img_dir = Path(self.source_structure["_source_img_dir"])
        self._source_caption_data_dir = Path(os.path.join(self.source_structure["_source_caption_dir"], "captions"))
        self._source_caption_index_dir = Path(os.path.join(self.source_structure["_source_caption_dir"], "index"))
        self._source_embd_data_dir = Path(os.path.join(self.source_structure["_source_embd_dir"]))
        self._source_embd_index_dir = Path(os.path.join(self.source_structure["_source_embd_dir"], "index"))

        # set up the target dir
        self.basepath = Path(os.path.join(self.source_structure["target_dir"], self.source_structure["dataset_name"]))

        # set up initial dirs
        for key, value in generic_dataset_structure.items():
            os.makedirs(os.path.join(self.basepath, value), exist_ok=True)
            self.__dict__[key] = Path(os.path.join(self.basepath, value))

        # set up hasher
        self.hashlength_dir = 15
        self.hash = hashlib.md5()

        # set up structure to remember what got hashed where
        self.hashed_directories = {}
        self.hashed_cap_tensors = {}
        self.hashed_embd_tensors = {}
        self.hashed_cap_jsons = {}
        self.hashed_embd_jsons = {}
        self.hashed_images = {}

        self.failed_images = []
        self.samples = {}

        # hash dir names
        self.hash_data_subdirs()

        for dirname, hashname in self.hashed_directories.items():
            self.hash_caption_tensors(dirname, hashname)
            self.hash_embd_tensors(dirname, hashname)
            self.hash_images(dirname, hashname)

    def hash_images(self, dirname, hashname):
        allfiles = Path(os.path.join(self._source_img_dir, dirname)).glob("*")
        for file in allfiles:
            try:
                img = Image.open(file)
                img.load()

                hash_filename = self.hash_file(file)
                future_imgfile = os.path.join(self._target_img_subdir, hashname, f"{hash_filename}{file.suffix}")
                shutil.copyfile(file, future_imgfile)
                self.hashed_images[file] = hash_filename
                if hash_filename in self.samples.keys():
                    raise IndexError(f"Hash collision for file: {file}")
                else:
                    self.samples[hash_filename] = Sample(id=hash_filename,
                                                         parent=dirname,
                                                         img_path=future_imgfile,
                                                         embd_tensor_path=os.path.join(
                                                             self._target_embd_knn_tensor_subdir,
                                                             f"{hashname}_embd_tensors.pt"),
                                                         cap_tensor_path=os.path.join(
                                                             self._target_cap_knn_tensor_subdir,
                                                             f"{hashname}_caption_tensors.pt"),
                                                         )

            except PIL.UnidentifiedImageError:
                self.failed_images.append(file)

    def hash_caption_json_files(self, dirname, hashname):
        prev_cap_jsonfile = os.path.join(self._source_caption_data_dir, dirname, "tensors.pt")
        future_cap_jsonfile = os.path.join(self._target_cap_knn_tensor_subdir, f"{hashname}_caption_jsons.json")

        self.hashed_cap_jsons[prev_cap_jsonfile] = future_cap_jsonfile

    def hash_caption_tensors(self, dirname, hashname):
        prev_cap_tensorfile = os.path.join(self._source_caption_data_dir, dirname, "tensors.pt")
        future_cap_tensorfile = os.path.join(self._target_cap_knn_tensor_subdir, f"{hashname}_caption_tensors.pt")
        shutil.copyfile(prev_cap_tensorfile, future_cap_tensorfile)
        self.hashed_cap_tensors[prev_cap_tensorfile] = future_cap_tensorfile

    def hash_embd_tensors(self, dirname, hashname):
        prev_embd_tensorfile = os.path.join(self._source_embd_data_dir, dirname, "tensors.pt")
        future_embd_tensorfile = os.path.join(self._target_embd_knn_tensor_subdir, f"{hashname}_embd_tensors.pt")
        shutil.copyfile(prev_embd_tensorfile, future_embd_tensorfile)
        self.hashed_embd_tensors[prev_embd_tensorfile] = future_embd_tensorfile

    def hash_data_subdirs(self):
        dirs_to_hash = [p for p in Path(self._source_img_dir).iterdir() if p.is_dir()]
        for datadir in dirs_to_hash:
            hashdir = self.hash_directory_name(datadir)
            print(f"Hashing directory {datadir.name} to {hashdir}")
            self.hashed_directories[datadir.name] = hashdir
            os.makedirs(os.path.join(self._target_img_subdir, hashdir), exist_ok=True)

    def hash_directory_name(self, pathlike):
        self.hash.update(pathlike.name.encode('utf-8'))
        return str(self.hash.hexdigest())[:self.hashlength_dir]

    def hash_file(self, imgpath):
        name = imgpath.name
        size = os.path.getsize(imgpath)
        self.hash.update((name + str(size)).encode('utf-8'))
        return str(self.hash.hexdigest())

    def serialize(self):
        output = {}
        for attr, state in vars(self).items():
            if isinstance(attr, Path):
                print(f"whoops! unexpected for {attr}")

            elif attr == "samples":     # not pretty but should work
                valueset = [sample.serialize() for sample in state.values()]
                output[attr] = dict(zip(state.keys(), valueset))

            elif isinstance(state, dict):
                keyset = [key.as_posix() if isinstance(key, Path) else key for key in state.keys()]
                valueset = [value.as_posix() if isinstance(value, Path) else value for value in state.values()]
                output[attr] = dict(zip(keyset, valueset))

            elif isinstance(state, Path):
                output[attr] = str(state.as_posix())
            elif isinstance(state, _hashlib.HASH):
                continue
            elif isinstance(state, Sample):
                continue
            else:
                output[attr] = state
        return dict(sorted(output.items()))


hash = HashSetTransformer(celador_test_input_structure)
with open("/home/frank/hashset_output.json", 'w') as jsonout:
    a = hash.serialize()
    json.dump(a, jsonout)
