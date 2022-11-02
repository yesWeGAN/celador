class Sample():
    def __init__(self,
        id = None, # potentially hash in the future
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

