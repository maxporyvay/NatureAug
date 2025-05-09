import itertools
import json
import sys

CLASSIFIER = "alexnet"
DATASET = "belgium_tsc"
LOSS_FN = "cross_entropy_loss"
NUM_EPOCHS = 10
OPTIMIZER = "sgd"
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.005
MOMENTUM = 0.9


if __name__ == "__main__":
    assert len(sys.argv) == 3
    assert sys.argv[1] == '--config-path'
    config_path = sys.argv[2]

    augmentations_lists = [
        [
            {
                "name": weather_aug_name,
                "params": {
                    "p": 0.5
                }
            },
            {
                "name": camera_aug_name,
                "params": {
                    "p": 0.5
                }
            }
        ]
        for weather_aug_name, camera_aug_name in itertools.product(
            [
                "fog_from_albumentations",
                "rain_from_albumentations",
                "shadow_from_albumentations",
                "snow_from_albumentations",
                "sun_flare_from_albumentations",
            ],
            [
                "defocus_blur_from_albumentations",
                "motion_blur_from_albumentations",
            ]
        )
    ]

    config = {
        "subexps": [
            {
                "name": f"exp{i + 1}",
                "dataset": {
                    "name": DATASET
                },
                "classifier": {
                    "name": CLASSIFIER
                },
                "training": {
                    "num_epochs": NUM_EPOCHS,
                    "optimizer": {
                        "name": OPTIMIZER,
                        "params": {
                            "lr": LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                            "momentum": MOMENTUM
                        }
                    },
                    "loss_function": {
                        "name": LOSS_FN
                    }
                },
                "augmentations": augmentations_lists[i]
            }
            for i in range(len(augmentations_lists))
        ]
    }

    with open(config_path, "w") as config_file:
        json.dump(config, config_file, indent=4)
