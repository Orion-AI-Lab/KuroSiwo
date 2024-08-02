# [Kuro Siwo: A global multi-temporal SAR dataset for rapid flood mapping](https://arxiv.org/abs/2311.12056)

  #### Latest updates:
    - [✔️] More events outside of Europe (43 in total)
    - [✔️] We included the respective SLC products and cropped patches in Kuro Siwo
    - [✔️] Downloading script and links have been updated for the new version
    - [✔️] Preprocessing pipelines for both GRD and SLC data can be found in `configs/`
    - [✔️] Updated paper: https://arxiv.org/abs/2311.12056
    - [ ] TODO: minor updates to training and dataloading code 

![Kuro Siwo](imgs/kuro_spatial.png)


# Table of Contents
- [Download the dataset](#download-kuro-siwo)
- [Data preprocessing](#data-preprocessing)
- [Repository structure](#kuro-siwo-repo-structure)
- [Pretrained models](#pretrained-models)
- [Citation](#citation)
### Download Kuro Siwo
  #### GRD Data
- The Kuro Siwo GRD Dataset can be downloaded either:
  - from the following [link](https://www.dropbox.com/scl/fo/xc69aclh0q4lykd22ynkb/AAaDu8gBtoSdOpmffv7JY50?rlkey=uds2b2aot6oubc9hmnrm7myy7&st=21u41kwx&dl=0),


  - or by executing ```scripts/download_kuro_siwo.sh```. This script will download and prepare the Kuro Siwo GRDD dataset for deep learning.

    #### Usage 

    1. Make sure to grant the necessary rights by executing `chmod +x scripts/download_kuro_siwo.sh`
    2. Execute `scripts/download_kuro_siwo.sh DESIRED_DATASET_ROOT_PATH` e.g: `./download_kuro_siwo.sh KuroRoot`
   
#### SLC Data
  - The SLC Preprocessed products can be downloaded from the following [link](https://www.dropbox.com/scl/fo/kknf6ycz6ywffopjxroys/AOIedl2NgWnOXQBEDUGv4m0?rlkey=rb18w8rzpwitg2w3nlhzklnyy&st=p1vv516h&dl=0).

  - Similarly, the cropped SLC patches (224x224 pixels) can be acquired from the following [link](https://www.dropbox.com/scl/fo/6u1bhbhd34rnn0u47o8dj/AK9vblAzDWqhPTqYvioPUb8?rlkey=i7k862563n936akuqlsdf3w66&st=0f7q3vno&dl=0).  


### Data preprocessing

The preprocessing pipelines used to generate the GRD and SLC products can be found at `configs/grd_preprocessing.xml` and `configs/slc_preprocessing.xml` repsectively.
### Kuro Siwo repo structure 
  - Kuro Siwo uses the [black](https://github.com/psf/black) python formatter. To activate it install pre-commit, running `pip install pre-commit`
and execute `pre-commit install`.
  - Training starts by running `python main.py`. The configurations are defined in the `configs` directory
 e.g 
    - model,
    - training pipeline 
      - Segmentation,
      - change detection
    - hyperparameters
  - `main.py` supports command line arguments that override the config files.
     e.g 
      ```
         python main.py --method=unet --backbone=resnet18 --dem=True --slope=False --batch_size=32
      ```


### Pretrained models
The weights of the top performing models can be accessed using the following links:
  - [FloodViT](https://www.dropbox.com/scl/fi/srw7u4cw1gtxrf4xzmsh7/floodvit.pt?rlkey=snskpq1qrdav5u2jya8k2bocg&dl=0)
  - [SNUNet](https://www.dropbox.com/scl/fi/3vlsveoobqe1wc71s5z2d/best_segmentation.pt?rlkey=xpy2thmozzxfzymr8b13m7n51&dl=0)

### Citation
If you use this work please cite:
```
@misc{bountos2024kurosiwo33billion,
      title={Kuro Siwo: 33 billion $m^2$ under the water. A global multi-temporal satellite dataset for rapid flood mapping}, 
      author={Nikolaos Ioannis Bountos and Maria Sdraka and Angelos Zavras and Ilektra Karasante and Andreas Karavias and Themistocles Herekakis and Angeliki Thanasou and Dimitrios Michail and Ioannis Papoutsis},
      year={2024},
      eprint={2311.12056},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.12056}, 
}
```
