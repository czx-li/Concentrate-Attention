# Concentrate Attention: Towards Domain-Generalizable Prompt Optimization for Language Models
This repository contains code for *Concentrate Attention: Towards Domain-Generalizable Prompt Optimization for Language Models* (https://web3.arxiv.org/abs/2406.10584, NeurIPS 2024) by Chengzhengxu Li, Xiaoming Liu*, Zhaohan Zhang, Yichen Wang, Chen Liu, Yu Lan, Chao Shen. 

In this codebase we conduct pilot experiments and find that (i) Prompts gaining more attention weight from PLMs’ deep layers are more generalizable and (ii) Prompts with more stable attention distributions in PLMs’ deep layers are more generalizable. 

![](figure1.png)

Thus, we offer a fresh objective towards domain-generalizable prompts optimization named ''Concentration'', which represents the ''lookback'' attention from the current decoding token to the prompt tokens, to increase the attention strength on prompts and reduce the fluctuation of attention distribution. 

![](figure0.png)

We adapt this new objective to popular soft prompt and hard prompt optimization methods, respectively. Experiments demonstrate that our idea improves comparison prompt optimization methods by 1.42% for soft prompt generalization and 2.16% for hard prompt generalization in accuracy on the multi-source domain generalization setting, while maintaining satisfying in-domain performance. 

# Setting Up

Our codebase requires the following Python and PyTorch versions: 
* Python >= 3.8
* PyTorch >= 1.8.1 (install from the [official website](https://pytorch.org/get-started/locally/))

Install our core modules with
```
git clone https://github.com/czx-li/Concentrate-Attention.git
```
Train and save our modules
```
python main.py
```
## Citation

If you find our work helpful, please cite us with the following BibTex entry:

```
@inproceedings{NEURIPS2024_061d5d1b,
 author = {Li, Chengzhengxu and Liu, Xiaoming and Zhang, Zhaohan and Wang, Yichen and Liu, Chen and Lan, Yu and Shen, Chao},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
 pages = {3391--3420},
 publisher = {Curran Associates, Inc.},
 title = {Concentrate Attention: Towards Domain-Generalizable Prompt Optimization for Language Models},
 url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/061d5d1b7d97117764f205d4e038f9eb-Paper-Conference.pdf},
 volume = {37},
 year = {2024}
}

```

Link to NeurIPS 2024 version paper: 
