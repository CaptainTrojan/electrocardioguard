# ElectroCardioGuard

This repository will contain all code, models and datasets relevant to ElectroCardioGuard (https://arxiv.org/abs/2306.06196).

## Installation

```
pip install -r requirements.txt
```

To recreate experiments, please setup a MLflow tracking server according to instructions in `dgn_mlflow_logger`.

## Getting started

Scripts for converting datasets to our HDF5 format are located in `dataset_conversion_scripts` directory. Scripts for downloading will be published shortly.

`results` contains full result tables, whose shortened and compact versions are published in our paper.

To run a single instance of grid search (model configuration), run `python pt_grid_search_instance.py` with corresponding arguments.

To evaluate our model in the gallery/probe matching task or overseer simulation task, run `pt_evaluate_as_classifier.py` with corresponding arguments. You can run `gallery_probe_all.sh` and `overseer_simulation_dev/test.sh` to directly reproduce our results, or update the model path parameter to evaluate a different model.
