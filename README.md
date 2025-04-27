# NatureAug: Natural image augmentation methods library

NatureAug is library providing access to making experiments with images classifiers using different natural augmentation techniques. Natural augmentation methods are divided into weather artifacts and camera artifacts augmentation methods.

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

## Classifiers

TBD

## Experiment configuration

TBD

## Running demo script

```
python3 demo.py --config-path <path_to_your_config>
```
Config example can be found in `config_example.json` file.
