# NatureAug: Natural image augmentation methods library

NatureAug is library providing access to making experiments with image classifiers using different natural augmentation techniques easily. Natural augmentation methods are divided into weather artifacts and camera artifacts augmentation methods.

## Augmentations

Weather artifacts based augmentations being supported now:
- Clouds (from [Imgaug](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_weather.html#imgaug.augmenters.weather.Clouds))
- Fog (from [Albumentations](https://albumentations.ai/docs/api-reference/augmentations/transforms/#RandomFog)/[Imgaug](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_weather.html#imgaug.augmenters.weather.Fog))
- Frost (from [Imgaug](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_imgcorruptlike.html#imgaug.augmenters.imgcorruptlike.Frost))
- Rain (from [Albumentations](https://albumentations.ai/docs/api-reference/augmentations/transforms/#RandomRain)/[Imgaug](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_weather.html#imgaug.augmenters.weather.Rain))
- Shadow (from [Albumentations](https://albumentations.ai/docs/api-reference/augmentations/transforms/#RandomShadow))
- Snow (from [Albumentations](https://albumentations.ai/docs/api-reference/augmentations/transforms/#RandomSnow)/[Imgaug](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_weather.html#imgaug.augmenters.weather.Snowflakes))
- Sun flare (from [Albumentations](https://albumentations.ai/docs/api-reference/augmentations/transforms/#RandomSunFlare))

Camera artifacts based augmentations being supported now:
- Defocus blur (from [Albumentations](https://albumentations.ai/docs/api-reference/augmentations/blur/transforms/#Defocus)/[Imgaug](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_imgcorruptlike.html#imgaug.augmenters.imgcorruptlike.DefocusBlur))
- Motion blur (from [Albumentations](https://albumentations.ai/docs/api-reference/augmentations/blur/transforms/#MotionBlur)/[Imgaug](https://imgaug.readthedocs.io/en/latest/source/api_augmenters_imgcorruptlike.html#imgaug.augmenters.imgcorruptlike.MotionBlur))

## Datasets

Two datasets are being supported now: [BelgiumTSC](https://btsd.ethz.ch/shareddata/) and [GTSRB](https://benchmark.ini.rub.de/gtsrb_dataset.html).

To download data from needed dataset, run appropriate bash script:
```
cd datasets
./belgium_tsc.sh
```

to download BelgiumTSC data, or
```
cd datasets
./gtsrb.sh  `
```
to download GTSRB data.

## Training process

Four classifiers from `torchvision.models` are being supported now:
- [alexnet](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.alexnet.html#torchvision.models.alexnet)
- [resnet16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18)
- [vgg11](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg11.html#torchvision.models.vgg11)
- [inception_v3](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.inception_v3.html#torchvision.models.inception_v3)

Two optimizers from `torch.optim` are being supported now:
- [SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD)
- [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html#torch.optim.Adam)

One loss function from `torch.nn` is being supported now:
- [Cross-entropy loss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

Two metrics from `torcheval.metrics` are being supported now:
- [Accuracy](https://docs.pytorch.org/torcheval/stable/generated/torcheval.metrics.functional.multiclass_accuracy.html#torcheval.metrics.functional.multiclass_accuracy): `micro` and `macro`
- [F1-score](https://docs.pytorch.org/torcheval/stable/generated/torcheval.metrics.functional.multiclass_f1_score.html#torcheval.metrics.functional.multiclass_f1_score): `macro`

## Experiment configuration

TBD

## Running demo script

```
python3 demo.py --config-path <path_to_your_config>
```
Config examples can be found in `config_examples` directory.

## Extending library with your own classes

### New augmentations

TBD

### New datasets

TBD

### New classifiers

TBD

### New optimizers

TBD

### New loss functions

TBD

### New metrics

TBD
