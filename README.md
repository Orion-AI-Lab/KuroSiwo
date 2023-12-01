# [Kuro Siwo: A global multi-temporal SAR dataset for rapid flood mapping](https://arxiv.org/abs/2311.12056) 

![Kuro Siwo](imgs/kuro_spatial.png)
### Download Kuro Siwo
- The Kuro Siwo Dataset can be downloaded either from the following [link](https://www.dropbox.com/scl/fo/nkqaa9se5zl3yng4bdai4/h?rlkey=bro222cvgu4lo3b4towo6gbmm&dl=0) or by executing:


- ```./download_kuro_siwo.sh```. This script will download and prepare the Kuro Siwo dataset for deep learning.

    #### Usage 

    1. Make sure to grant the necessary rights by executing `chmod +x download_kuro_siwo.sh`
    2. Execute `./download_kuro_siwo.sh DESIRED_DATASET_ROOT_PATH` e.g: `./download_kuro_siwo.sh .`
   

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