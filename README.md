# RAH-KBQA [EMNLP 2023]
This is the code for the EMNLP 2023 Findings paper: [Relation-Aware Question Answering for Heterogeneous Knowledge Graphs](to be continue).


## Overview 
Our methods improves instruction decoding and execution for KGQA via adaptive reasoning, as shown:

![](./pipeline.pdf)


## Get Started
We have simple requirements in `requirements.txt'. You can always check if you can run the code immediately.

We use the pre-processed data from: https://drive.google.com/drive/folders/1qRXeuoL-ArQY7pJFnMpNnBu0G-cOz6xv
Download it and extract it to a folder named "data".

__Acknowledgements__: 

[NSM](https://github.com/RichardHGL/WSDM2021_NSM): Datasets (webqsp, CWQ, MetaQA) / Code.

[GraftNet](https://github.com/haitian-sun/GraftNet): Datasets (webqsp incomplete, MetaQA) / Code.

## Training

To run Webqsp:
```
python main.py ReaRev --entity_dim 128 --num_epoch 150 --batch_size 8 --eval_every 2 \ 
--data_folder data/webqsp/ --lm sbert --num_iter 3 --num_ins 2 --num_gnn 2 \
--relation_word_emb True --experiment_name Webqsp322 --name webqsp
```

To run CWQ:
```
python main.py ReaRev --entity_dim 128 --num_epoch 70 --batch_size 8 --eval_every 2 \
--data_folder data/CWQ/ --lm sbert --num_iter 2 --num_ins 3 --num_gnn 3 \
--relation_word_emb True --experiment_name CWQ --name cwq
```


For incomplete Webqsp, see  'data/incomplete/' (after obtaining them by [GraftNet](https://github.com/haitian-sun/GraftNet)). If you cannot afford a lot of memory for CWQ, use the '--data_eff' argument (see our arguments in `parsing.py').

## Results



|   Models    |  Webqsp  |   CWQ    | 
|:-----------:|:--------:|:--------:|
|   KV-Mem    |   46.7   |   21.1   | 
|  GraftNet   |   66.4   |   32.8   |
|   PullNet   |   68.1   |   45.9   | 
| NSM-distill |   74.3   |   48.8   | 
|   ReaRev    |   76.4   |   52.9   | 
|  RAH-KBQA   | **77.2** | **54.4** | 

## Cite
If you find our code or method useful, please cite our work as
```
to be continue
```
or
```
```
