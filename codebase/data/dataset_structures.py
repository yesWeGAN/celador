generic_dataset_structure = {"_target_img_subdir": "img",
                             "_target_knn_subdir": "knn",
                             "_target_t5_subdir": "t5",
                             "_target_cap_knn_tensor_subdir": "knn/caption/tensors",
                             "_target_embd_knn_tensor_subdir": "knn/embd/tensors",
                             "_target_cap_knn_index_subdir": "knn/caption/index",
                             "_target_embd_knn_index_subdir": "knn/embd/index",
                             "_target_combi_knn_index_subdir": "knn/combi/index",
                             "_target_combi_knn_tensor_subdir": "knn/combi/tensor"
                             }

celador_input_structure = {"_source_caption_dir": "/home/frank/data/celador/caption",
                           "_source_embd_dir": "/home/frank/data/celador/knn",
                           "_source_img_dir": "/home/frank/data/celador/img/bing",
                           "sourcepath": "",
                           "dataset_name": "",
                           "target_dir": ""
                           }

celador_test_input_structure = {"_source_caption_dir": "/home/frank/Documents/dev_data_subset/celador/caption",
                                "_source_embd_dir": "/home/frank/Documents/dev_data_subset/celador/knn",
                                "_source_img_dir": "/home/frank/Documents/dev_data_subset/celador/img/bing",
                                "sourcepath": "/home/frank/Documents/dev_data_subset/celador",
                                "dataset_name": "celador_transform",
                                "target_dir": "/home/frank/Documents/dev_data_subset/target"
                                }
