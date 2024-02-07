# [Kuro Siwo: A global multi-temporal SAR dataset for rapid flood mapping](https://arxiv.org/abs/2311.12056)  (WIP)


![Kuro Siwo](imgs/kuro_spatial.png)

### Citation
If you use this work please cite:
```
@article{bountos2023kuro,
  title={Kuro Siwo: 12.1 billion $ m\^{} 2$ under the water. A global multi-temporal satellite dataset for rapid flood mapping},
  author={Bountos, Nikolaos Ioannis and Sdraka, Maria and Zavras, Angelos and Karasante, Ilektra and Karavias, Andreas and Herekakis, Themistocles and Thanasou, Angeliki and Michail, Dimitrios and Papoutsis, Ioannis},
  journal={arXiv preprint arXiv:2311.12056},
  year={2023}
}
```

### Download Kuro Siwo
- The Kuro Siwo Dataset can be downloaded either:
  - from the following [link](https://www.dropbox.com/scl/fo/nkqaa9se5zl3yng4bdai4/h?rlkey=bro222cvgu4lo3b4towo6gbmm&dl=0),


  - or by executing ```scripts/download_kuro_siwo.sh```. This script will download and prepare the Kuro Siwo dataset for deep learning.

    #### Usage 

    1. Make sure to grant the necessary rights by executing `chmod +x scripts/download_kuro_siwo.sh`
    2. Execute `scripts/download_kuro_siwo.sh DESIRED_DATASET_ROOT_PATH` e.g: `./download_kuro_siwo.sh KuroRoot`
   

#### Kuro Siwo repo structure 
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


#### Pretrained models
The weights of the top performing models can be accessed using the following links:
  - [FloodViT](https://www.dropbox.com/scl/fi/srw7u4cw1gtxrf4xzmsh7/floodvit.pt?rlkey=snskpq1qrdav5u2jya8k2bocg&dl=0)
  - [SNUNet](https://www.dropbox.com/scl/fi/3vlsveoobqe1wc71s5z2d/best_segmentation.pt?rlkey=xpy2thmozzxfzymr8b13m7n51&dl=0)