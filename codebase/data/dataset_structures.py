generic_dataset_structure = {"_target_img_subdir": "img",
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

dataset_structures = {"celador":
                          {"_source_caption_dir": "/home/frank/data/celador/caption",
                           "_source_embd_dir": "/home/frank/data/celador/knn",
                           "_source_img_dir": "/home/frank/data/celador/img/bing",
                           "sourcepath": "/home/frank/data/celador",
                           "dataset_name": "celador",
                           "target_dir": "/home/frank/datasets"
                           },

                      "celador_test":
                          {"_source_caption_dir": "/home/frank/Documents/dev_data_subset/celador/caption",
                           "_source_embd_dir": "/home/frank/Documents/dev_data_subset/celador/knn",
                           "_source_img_dir": "/home/frank/Documents/dev_data_subset/celador/img/bing",
                           "sourcepath": "/home/frank/Documents/dev_data_subset/celador",
                           "dataset_name": "celador_transform",
                           "target_dir": "/home/frank/Documents/dev_data_subset/target"
                           },

                      "hash-0":
                          {"_source_caption_dir": "/home/frank/ssd/backup/data/instagram/caption",
                           "_source_embd_dir": "/home/frank/ssd/backup/data/instagram/knn",
                           "_source_img_dir": "/home/frank/ssd/backup/data/instagram/bogan/____00subset",
                           "sourcepath": "/home/frank/ssd/backup/data/instagram",
                           "dataset_name": "hash",
                           "target_dir": "/home/frank/ssd/backup/datasets"
                           },

                      "hash-1":
                          {"_source_caption_dir": "/home/frank/ssd/backup/data/instagram/caption",
                           "_source_embd_dir": "/home/frank/ssd/backup/data/instagram/knn",
                           "_source_img_dir": "/home/frank/ssd/backup/data/instagram/bogan/____01subset",
                           "sourcepath": "/home/frank/ssd/backup/data/instagram",
                           "dataset_name": "hash",
                           "target_dir": "/home/frank/ssd/backup/datasets"
                           },
                      "ATF":
                          {"_source_caption_dir": "/home/frank/.ATF/caption",
                           "_source_embd_dir": "/home/frank/.ATF/knn",
                           "_source_img_dir": "/home/frank/.ATF/img",
                           "sourcepath": "/home/frank/.ATF",
                           "dataset_name": "atf",
                           "target_dir": "/home/frank/ssd/backup/datasets"
                           },
                      }
