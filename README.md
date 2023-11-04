# FGWMixup: Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications


This is the code for the paper: Fused Gromov-Wasserstein Graph Mixup for Graph-level Classifications, published in NeurIPS'23.

**Paper link** 🔗:

arXiv version: https://arxiv.org/abs/2306.15963 

OpenReview version: https://openreview.net/forum?id=uqkUguNu40&noteId=0qcp06CFB6

Thanks for your interest in our work! If our work helps, please don't forget to cite our paper!

### File Structure

- ```./src/```: source codes

  ```gmixup_dgl.py```: Main python file to run FGWMixup
  
  ```gromov_mixup.py```: Conducting mixup of two samples
  
  ```FGW_barycenter.py```: Calculating FGW barycenter and its accelerated version
  
  ```models_dgl.py```: GNN architectures
  
  ```utils_dgl.py```: Some utilities

- ```run_gmixup.sh```: sh command to run FGWMixup


### Requirements 
Suggested Enviornments:
- Python 3.9
- PyTorch 1.11.0
- DGL 1.0.2
- POT 0.8.2

