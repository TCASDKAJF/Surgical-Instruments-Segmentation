from box import Box

config = {
    "num_devices": 1,
    "batch_size": 12,
    "num_workers": 1,
    "num_epochs": 20,
    "eval_interval": 2,
    "out_dir": "/data/result/changxiu/open_source/SAM/model/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": "/data/result/changxiu/open_source/SAM/model/sam_vit_b_01ec64.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": "/data/result/changxiu/open_source/SAM/instrument_dataset/image/training",
            "annotation_file": "/data/result/changxiu/open_source/SAM/instrument_dataset/coco_format/annotations_training.json"
        },
        "val": {
            "root_dir": "/data/result/changxiu/open_source/SAM/instrument_dataset/image/validation",
            "annotation_file": "/data/result/changxiu/open_source/SAM/instrument_dataset/coco_format/annotations_validation.json"
        }
    }
}

cfg = Box(config)
