generic_dataset_structure = {
    "_target_img_subdir": "img",
    "_target_knn_subdir": "knn",
    "_target_t5_subdir": "t5",
    "_target_cap_knn_tensor_subdir": "knn/caption/tensors",
    "_target_embd_knn_tensor_subdir": "knn/embd/tensors",
    "_target_cap_knn_index_subdir": "knn/caption/index",
    "_target_embd_knn_index_subdir": "knn/embd/index",
    "_target_combi_knn_index_subdir": "knn/combi/index",
    "_target_combi_knn_tensor_subdir": "knn/combi/tensor",
    "_target_json_backup_dir": "dataset/backup",
    "_target_dataset_dir": "dataset",
}

dataset_structures = {
    "celador": {
        "_source_caption_dir": "/home/frank/data/celador/caption",
        "_source_embd_dir": "/home/frank/data/celador/knn",
        "_source_img_dir": "/home/frank/data/celador/img/bing",
        "sourcepath": "/home/frank/data/celador",
        "dataset_name": "celador",
        "target_dir": "/home/frank/datasets",
    },
    "celador_test": {
        "_source_caption_dir": "/home/frank/Documents/dev_data_subset/celador/caption",
        "_source_embd_dir": "/home/frank/Documents/dev_data_subset/celador/knn",
        "_source_img_dir": "/home/frank/Documents/dev_data_subset/celador/img/bing",
        "sourcepath": "/home/frank/Documents/dev_data_subset/celador",
        "dataset_name": "celador_transform",
        "target_dir": "/home/frank/Documents/dev_data_subset/target",
    },
    "hash-2": {
        "_source_caption_dir": "/home/frank/insta_extraction",
        "_source_embd_dir": "/home/frank/insta_extraction",
        "_source_img_dir": "/home/frank/insta_extraction/img",
        "sourcepath": "/home/frank/insta_extraction",
        "dataset_name": "add-hash",
        "target_dir": "/home/frank/datasets",
    },
}
