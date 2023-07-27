import _hashlib
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
import hashlib
import torch.utils.data
from PIL import Image
import PIL
from tqdm import tqdm
import traceback
import argparse
from codebase.data.sample import Sample
from codebase.data.dataset_structures import generic_dataset_structure, dataset_structures
import gc


def parse_args():
    """Parse arguments from command-line"""
    parser = argparse.ArgumentParser(description='Process dataset output into hashed dataset format.')
    parser.add_argument('-d', '--dataset', type=str, default="celador", help='dataset specification')
    parser.add_argument('-t', '--transform', action='store_true', help='transform dataset from raw')
    args = parser.parse_args()
    return args
"""TODO incorporate stuff to find hashes
from pprint import pprint

def search_hash_in_set(hashidlist):
    hashmatches = []
    for hashid in hashidlist:
        for key in hashset.samples.keys():
            if hashid in key:
                hashmatches.append(key)
    return hashmatches

def print_hashmatches(hashmatches):
    for match in hashmatches:
        pprint(vars(hashset.samples[match]))
    """

class HashSetTransformer:
    """This class combines a dataset with img, img embd, cap, cap embd in one access structure.
    Used to transform existing datasets into a common format."""

    def __init__(self,
                 source_structure: dict
                 ):
        # set up the source dirs
        self.source_structure = source_structure
        self.sourcepath = Path(source_structure["sourcepath"])
        self.target_dir = Path(self.source_structure["target_dir"])
        self._source_img_dir = Path(self.source_structure["_source_img_dir"])
        self._source_caption_data_dir = Path(os.path.join(self.source_structure["_source_caption_dir"], "captions"))
        self._source_caption_index_dir = Path(os.path.join(self.source_structure["_source_caption_dir"], "index"))
        self._source_embd_data_dir = Path(os.path.join(self.source_structure["_source_embd_dir"], "embd"))
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

        self.hashed_embd_index = {}  # key: idx in embd-index, value: hash
        self.hashed_cap_index = {}  # key: idx in caption-index, value: hash

        self.failed_images = []
        self.samples = {}

        # hash dir names
        self.hash_data_subdirs()
        self.copy = True

        for dirname, hashname in tqdm(self.hashed_directories.items()):
            try:
                # I can continue working here, the error is just missing tensors
                # (but they were also missing when the index was created)
                self.hash_caption_tensors(dirname, hashname)
                self.hash_embd_tensors(dirname, hashname)
                self.hash_images(dirname, hashname)
                # print(f"After hashing images, {len(self.failed_images)} failed.")
                self.hash_caption_json_files(dirname, hashname)
                self.hash_embd_json_files(dirname, hashname)

            except:
                traceback.print_exc()

        self.save()

    def hash_images(self, dirname, hashname):
        allfiles = Path(os.path.join(self._source_img_dir, dirname)).glob("*")
        for file in allfiles:
            try:
                if self.copy:
                    img = Image.open(file)
                    # img.load()        # hash dataset: unsafe operation trial

                hash_filename = self.hash_file(file)
                future_imgfile = os.path.join(self._target_img_subdir, hashname, f"{hash_filename}{file.suffix}")

                """if not os.path.isfile(future_imgfile):
                    self.failed_images.append(future_imgfile)
                    continue"""

                if not os.path.isfile(future_imgfile):
                    shutil.copyfile(file, future_imgfile)

                self.hashed_images[file] = hash_filename
                if hash_filename in self.samples.keys():
                    raise IndexError(f"Hash collision for file: {file}")
                else:
                    self.samples[hash_filename] = Sample(hashid=hash_filename,
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
        try:
            cap_jsondir = os.path.join(self._source_caption_data_dir, dirname)
            filep = next(Path(cap_jsondir).rglob("*captions.json"))
        except StopIteration:
            raise FileNotFoundError(f"No caption json file found for {dirname}. Exiting.")

        future_cap_jsonfile = os.path.join(self._target_cap_knn_tensor_subdir, f"{hashname}_caption_tensors.pt")
        jsondata = json.load(open(filep, 'r'))

        hash_to_caption_and_index = {}
        # in the caption file, there is the old path key and caption value. add each caption to respective hashID
        for k, key in enumerate(jsondata.keys()):
            try:
                caption = jsondata[key]
                hashfile = self.hashed_images[Path(key)]
                self.samples[hashfile].caption = caption
                self.samples[hashfile].cap_tensor_path = future_cap_jsonfile
                self.samples[hashfile].cap_tensor_index = k
                hash_to_caption_and_index[hashfile] = [caption, k]
            except KeyError:
                raise IndexError(f"Image not been hashed yet: {Path(key).name}")
        if self.copy:
            json.dump(hash_to_caption_and_index,
                      open(os.path.join(self._target_json_backup_dir, f"{hashname}_captions.json"), 'w'))

    def hash_embd_json_files(self, dirname, hashname):
        try:
            cap_jsondir = os.path.join(self._source_embd_data_dir, dirname)
            filep = next(Path(cap_jsondir).rglob("*paths.json"))
        except StopIteration:
            raise FileNotFoundError(f"No embd json file found for {dirname}. Exiting.")

        future_embd_jsonfile = os.path.join(self._target_embd_knn_tensor_subdir, f"{hashname}_embd_tensors.pt")
        jsondata = json.load(open(filep, 'r'))

        hash_to_embd_index = {}
        # in the caption file, there is the old path key and caption value. add each caption to respective hashID
        for k, key in enumerate(jsondata.keys()):
            try:
                imgpath = jsondata[key]
                if "instagram" in self.sourcepath.as_posix():
                    imgpath = os.path.join(self._source_img_dir, dirname, Path(imgpath).name)
                hashfile = self.hashed_images[Path(imgpath)]
                self.samples[hashfile].embd_tensor_path = future_embd_jsonfile
                self.samples[hashfile].embd_tensor_index = k
                hash_to_embd_index[hashfile] = k
            except KeyError:
                raise IndexError(f"Image hash not been hashed yet: {Path(key).name}")
        if self.copy:
            json.dump(hash_to_embd_index,
                      open(os.path.join(self._target_json_backup_dir, f"{hashname}_embd_indices.json"), 'w'))

    def hash_caption_tensors(self, dirname, hashname):
        prev_cap_tensorfile = os.path.join(self._source_caption_data_dir, dirname, "tensors.pt")
        future_cap_tensorfile = os.path.join(self._target_cap_knn_tensor_subdir, f"{hashname}_caption_tensors.pt")
        if self.copy:
            shutil.copyfile(prev_cap_tensorfile, future_cap_tensorfile)
        self.hashed_cap_tensors[prev_cap_tensorfile] = future_cap_tensorfile

    def hash_embd_tensors(self, dirname, hashname):
        prev_embd_tensorfile = os.path.join(self._source_embd_data_dir, dirname, "tensors.pt")
        future_embd_tensorfile = os.path.join(self._target_embd_knn_tensor_subdir, f"{hashname}_embd_tensors.pt")
        if self.copy:
            shutil.copyfile(prev_embd_tensorfile, future_embd_tensorfile)
        self.hashed_embd_tensors[prev_embd_tensorfile] = future_embd_tensorfile

    def hash_data_subdirs(self):
        dirs_to_hash = [p for p in Path(self._source_img_dir).iterdir() if p.is_dir()]
        for datadir in dirs_to_hash:
            hashdir = self.hash_directory_name(datadir)
            self.hashed_directories[datadir.name] = hashdir
            os.makedirs(os.path.join(self._target_img_subdir, hashdir), exist_ok=True)

    def hash_directory_name(self, pathlike):
        self.hash.update(pathlike.name.encode('utf-8'))
        return str(self.hash.hexdigest())[:self.hashlength_dir]

    def hash_file(self, imgpath: Path):
        name = imgpath.name
        size = os.path.getsize(imgpath)
        self.hash.update((name + str(size)).encode('utf-8'))
        return str(self.hash.hexdigest())

    def serialize(self):
        output = {}
        for attr, state in vars(self).items():
            if isinstance(attr, Path):
                print(f"whoops! unexpected for {attr}")

            elif attr == "samples":  # not pretty but should work
                valueset = [sample.serialize() for sample in state.values()]
                output[attr] = dict(zip(state.keys(), valueset))

            elif isinstance(state, dict):
                if len(state) == 0:
                    continue  # skip empty entries
                else:
                    keyset = [key.as_posix() if isinstance(key, Path) else key for key in state.keys()]
                    valueset = [value.as_posix() if isinstance(value, Path) else value for value in state.values()]
                    output[attr] = dict(zip(keyset, valueset))

            elif isinstance(state, list):
                if len(state) == 0:
                    continue  # skip empty entries
                else:
                    keyset = [key.as_posix() if isinstance(key, Path) else key for key in state]
                    output[attr] = keyset

            elif isinstance(state, Path):
                output[attr] = str(state.as_posix())
            elif isinstance(state, _hashlib.HASH):
                continue
            elif isinstance(state, Sample):
                continue
            else:
                output[attr] = state

        return dict(sorted(output.items()))

    def save(self):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        with open(os.path.join(self._target_dataset_dir, f"hashset_{date_time}.json"), 'w') as jsonout:
            json.dump(self.serialize(), jsonout)


class HashSet(torch.utils.data.Dataset):
    def __init__(self, datapath=None):
        try:
            jsonpath = next(Path(datapath).rglob("hashset*.json"))
        except StopIteration:
            raise FileNotFoundError(f"No dataset json file found in this directory. Exiting.")

        self.jsonfile = json.load(open(jsonpath, 'r'))
        for key, value in self.jsonfile.items():
            if not key == "samples":
                self.__dict__[key] = value
        self.samples = {}
        self.deserialize_samples()
        self.check_sample_completeness()
        self.hash_to_index_idx = {}
        self.idx_to_hash = {}
        self.index_inference_setup()
        self.jsonfile = None

    def index_inference_setup(self):
        """Prepare a mapping from knn-index to hash."""

        for sample in self.samples.values():
            self.idx_to_hash[sample.knn_idx] = sample.id

    def prepare_samplelist_from_query(self, query: list):
        """"""
        samplelist = []
        for q in query:
            try:
                samplelist.append(self.samples[q])
            except IndexError:
                print(f"Can't associate {q} with valid sample. Ignored in query.")
        return samplelist

    def process_result_list(self, resultlist: list):
        """Take a list of HashInferenceResult and fill in the attribute for samples from the dataset."""
        for k, result in enumerate(resultlist):
            resultlist[k].result_samples = [self.samples[self.idx_to_hash[idx]] for idx in result.result_indices]
        return resultlist

    def process_result(self, result):
        """Take the HashInferenceResult and fill in the attribute for samples from the dataset."""
        result.result_samples = [self.samples[self.idx_to_hash[idx]] for idx in result.result_indices]
        return result

    def deserialize_samples(self):
        for key, entry in self.jsonfile["samples"].items():
            sample = Sample(hashid=entry["id"],
                            parent=entry["parent"],
                            img_path=entry["img_path"],
                            embd_tensor_path=entry["embd_tensor_path"],
                            embd_tensor_index=entry["embd_tensor_index"],
                            # embd_index_idx=entry["embd_index_idx"],
                            caption=entry["caption"],
                            cap_tensor_path=entry["cap_tensor_path"],
                            cap_tensor_index=entry["cap_tensor_index"],
                            # cap_index_idx=entry["cap_index_idx"],
                            classification=entry["classification"],
                            knn_idx=entry["knn_idx"])
            self.samples[sample.id] = sample

    def serialize(self):
        output = {}
        for attr, state in vars(self).items():
            if isinstance(attr, Path):
                print(f"whoops! unexpected for {attr}")

            elif attr == "jsonfile":  # otherwise file doubles in size everytime!
                continue

            elif attr == "samples":  # not pretty but should work
                valueset = [sample.serialize() for sample in state.values()]
                output[attr] = dict(zip(state.keys(), valueset))

            elif isinstance(state, dict):
                if len(state) == 0:
                    continue  # skip empty entries
                else:
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

    def save(self):
        now = datetime.now()
        date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
        with open(os.path.join(self._target_dataset_dir, f"hashset_{date_time}.json"), 'w') as jsonout:
            json.dump(self.serialize(), jsonout)

    def check_sample_completeness(self):
        complete_samples = []
        for key, sample in self.samples.items():
            if sample.embd_tensor_index is not None and sample.cap_tensor_index is not None:
                complete_samples.append(sample)
        complete_samples = sorted(complete_samples, key=lambda x: x.embd_tensor_path)
        ordered_samples = {sample.id: sample for sample in complete_samples}
        self.samples = ordered_samples

    def load_tensors(self):
        embd_tensors = []
        cap_tensors = []
        current_sample = next(iter(self.samples.values()))
        current_embd_tensorpath = current_sample.embd_tensor_path
        current_embd_tensor = torch.load(open(current_sample.embd_tensor_path, 'rb')).cpu()
        current_cap_tensor = torch.load(open(current_sample.cap_tensor_path, 'rb')).cpu()

        for k, (hashid, sample) in enumerate(self.samples.items()):

            if k != len(embd_tensors):
                print(f"Diverging index and tensor length: k {k} and {len(embd_tensors)}")
                sys.exit(1)

            if sample.embd_tensor_path == current_embd_tensorpath:
                if len(current_embd_tensor.shape) < 2 or len(current_cap_tensor.shape) < 2:
                    embd_tensors.append(current_embd_tensor.squeeze())
                    cap_tensors.append(current_cap_tensor.squeeze())
                    self.hash_to_index_idx[sample.id] = k
                    self.samples[sample.id].knn_idx = k

                else:
                    embd_tensors.append(current_embd_tensor[int(sample.embd_tensor_index)].squeeze())
                    cap_tensors.append(current_cap_tensor[int(sample.cap_tensor_index)].squeeze())
                    self.hash_to_index_idx[sample.id] = k
                    self.samples[sample.id].knn_idx = k
            else:
                current_embd_tensorpath = sample.embd_tensor_path
                current_embd_tensor = torch.load(open(sample.embd_tensor_path, 'rb')).cpu()
                current_cap_tensor = torch.load(open(sample.cap_tensor_path, 'rb')).cpu()
                if len(current_embd_tensor.shape) < 2 or len(current_cap_tensor.shape) < 2:
                    embd_tensors.append(current_embd_tensor.squeeze())
                    cap_tensors.append(current_cap_tensor.squeeze())
                    self.hash_to_index_idx[sample.id] = k
                    self.samples[sample.id].knn_idx = k
                else:
                    embd_tensors.append(current_embd_tensor[int(sample.embd_tensor_index)].squeeze())
                    cap_tensors.append(current_cap_tensor[int(sample.cap_tensor_index)].squeeze())
                    self.hash_to_index_idx[sample.id] = k
                    self.samples[sample.id].knn_idx = k

        assert len(cap_tensors) == len(embd_tensors), "embd and caption tensors are not of same lenght!"

        embd_cat = torch.stack(embd_tensors)
        del embd_tensors
        torch.save(embd_cat, os.path.join(self._target_embd_knn_tensor_subdir, "embd_vectors_combined.pt"))
        gc.collect()
        cap_cat = torch.stack(cap_tensors)
        del cap_tensors
        torch.save(cap_cat, os.path.join(self._target_cap_knn_tensor_subdir, "caption_vectors_combined.pt"))
        gc.collect()

        combi_cat = torch.cat([embd_cat, cap_cat], dim=1)
        torch.save(combi_cat, os.path.join(self._target_combi_knn_tensor_subdir, "combivector.pt"))

        assert int(combi_cat.shape[1]) == int(embd_cat.shape[1] + cap_cat.shape[
            1]), f"Img/cap tensors concatenated in wrong dimension. Shape: {combi_cat.shape}"

        return embd_cat.cpu().numpy(), cap_cat.cpu().numpy(), combi_cat.cpu().numpy()

    def incorporate_labels(self, labelfile):
        pass

    def __len__(self):
        return len(self.samples.keys())

    def __getitem__(self, item):
        pass


def main():
    """Transform a given dataset to HashSet format. Specify data"""
    inputs = parse_args()
    if inputs.transform:
        hashsettransformation = HashSetTransformer(dataset_structures[inputs.dataset])
    dataset_path = os.path.join(dataset_structures[inputs.dataset]["target_dir"], dataset_structures[inputs.dataset]["dataset_name"], "dataset")
    hashset = HashSet(dataset_path)
    hashset.load_tensors()
    hashset.save()


if __name__ == '__main__':
    main()
