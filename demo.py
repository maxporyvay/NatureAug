import json
import sys

from natureaug.experiment import full_experiment_pipeline


if __name__ == '__main__':
    assert len(sys.argv) == 3
    assert sys.argv[1] == '--config-path'
    config_path = sys.argv[2]
    with open(config_path) as config_file:
        config = json.load(config_file)

    full_experiment_pipeline(config)
