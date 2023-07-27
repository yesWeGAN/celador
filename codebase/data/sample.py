import os
import torch
from PIL import Image
from pathlib import Path


class Sample:
    """Class to hold all information about a sample.
    TODO: add a collate_fn() to retrieve infos.
    """

    def __init__(self,
                 hashid: str = None,  # hash
                 parent: str | Path = None,  # original folder name
                 img_path: str | Path = None,  # current
                 embd_tensor_path: str | Path = None,
                 embd_tensor_index: str | Path = None,
                 embd_index_idx: int = None,
                 caption: str = None,
                 cap_tensor_path: str | Path = None,
                 cap_tensor_index: str | Path = None,
                 cap_index_idx: int = None,
                 classification: str = None,
                 knn_idx: int = None,
                 ) -> object:
        """

        :param hashid: unique hash-id
        :param parent: original folder name (user, bing query, ..)
        :param img_path: current image path (hashed)
        :param embd_tensor_path: path to the embedding tensor
        :param embd_tensor_index: path to the embedding index file
        :param embd_index_idx: index in embedding tensor
        :param caption: the caption for the image
        :param cap_tensor_path: path to the caption tensor
        :param cap_tensor_index: path to the caption index
        :param cap_index_idx: index in the caption tensor
        :param classification: classification string (0-9 classes)
        """
        self.id = hashid
        self.parent = parent
        self.img_path = img_path
        self.embd_tensor_path = embd_tensor_path
        self.embd_tensor_index = embd_tensor_index
        self.caption = caption
        self.cap_tensor_path = cap_tensor_path
        self.cap_tensor_index = cap_tensor_index
        self.classification = classification
        self.combi_tensor_path = os.path.join(Path(self.embd_tensor_path).parent.as_posix().replace("embd", "combi").replace("tensors", "tensor"), "combivector.pt")
        self.knn_idx = knn_idx  # attribute is new

    def serialize(self):
        return vars(self)

    def vis_representation(self):
        """Returns the representation used in visualization module."""
        textboxstr = '\n'.join([
            self.caption,
            self.classification if self.classification else "unlabeled",
        ])
        image = Image.open(self.img_path)
        return {"textboxstr": textboxstr, "image": image, "id": self.id}

    def check_self_complete(self):
        """Checks if all attributes are filled. Exception for classification."""
        for attr, val in vars(self).items():
            unchecked_args = [(fragment in attr) for fragment in ["classification", "idx"]]
            if any(unchecked_args):
                continue
            if attr:
                continue
            if self.embd_tensor_index is None:
                raise AttributeError(f"{attr} is None for {self.id}.")

    def get_embedding(self, embeddingtype="both"):
        tensor = None
        if embeddingtype == "embd":
            tensor = torch.load(open(self.embd_tensor_path, "rb"))[self.embd_tensor_index]
        if embeddingtype == "caption":
            tensor = torch.load(open(self.cap_tensor_path, "rb"))[self.cap_tensor_index]
        if embeddingtype == "both":
            tensor = torch.load(open(self.combi_tensor_path, 'rb'))[self.knn_idx]

        return torch.tensor(tensor).unsqueeze(dim=0)
