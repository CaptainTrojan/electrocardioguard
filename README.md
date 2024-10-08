# ElectroCardioGuard

This repository contains all code and models relevant to ElectroCardioGuard (https://www.sciencedirect.com/science/article/pii/S0950705123007645, https://arxiv.org/abs/2306.06196).

Due to the large size of the IKEM dataset, we opted to publish it on Zenodo (https://zenodo.org/record/8393007).

## Installation

```
pip install -r requirements.txt
```

To recreate experiments, please setup a MLflow tracking server according to instructions in `dgn_mlflow_logger`.

## Getting started

Scripts for converting datasets to our HDF5 format are located in `dataset_conversion_scripts` directory. In order to download all datasets and apply our pre-processing/compression to them, call

```
python download_datasets.py
```

The script will take a long time to fully complete (4-7 hours). To verify the process completed successfully, you can run `python dataset_stats.py` and compare the numbers with this table:

| Title    | № ECGs | № patients | Size  |
|----------|--------|------------|-------|
| PTB      | 549    | 290        | 69MB  |
| PTB-XL   | 21,799 | 18,869     | 2.1GB |
| CODE-15% | 345,106| 233,479    | 22GB  |

In order to save disk space, we discard redundant leads (III, aVF, aVR, aVL). Specifically, the remaining leads V1-6, I, and II are stored in this order respectively from indices 0 to 7. We also quantize voltages to a 16-bit scale with 4.88 μV per bit. Tracings are stored in HDF5 files with a single tracing per chunk for fast random access.

`results` contains full result tables, whose shortened and compact versions are published in our paper.

To run a single instance of grid search (model configuration), run `python pt_grid_search_instance.py` with corresponding arguments.

To evaluate our model in the gallery/probe matching task or overseer simulation task, run `pt_evaluate_as_classifier.py` with corresponding arguments. You can run `gallery_probe_all.sh` and `overseer_simulation_dev/test.sh` to directly reproduce our results, or update the model path parameter to evaluate a different model.

Our model is built on top of CDIL-CNN (`models/pt_cdil_cnn.py`). The original implementation can be found here: https://github.com/LeiCheng-no/CDIL-CNN.

If you've found our work useful, please cite our publication:

```
@article{sejak2023electrocardioguard,
  title={ElectroCardioGuard: Preventing patient misidentification in electrocardiogram databases through neural networks},
  author={Sej{\'a}k, Michal and Sido, Jakub and {\v{Z}}ahour, David},
  journal={Knowledge-Based Systems},
  volume={280},
  pages={111014},
  year={2023},
  publisher={Elsevier}
}
```
