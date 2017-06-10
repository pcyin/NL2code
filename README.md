# NL2code

A syntactic neural model for parsing natural language to executable code [paper](https://arxiv.org/abs/1704.01696). 

## Dataset and Trained Models

Get serialized datasets and trained models from [here](https://drive.google.com/drive/folders/0B14lJ2VVvtmJWEQ5RlFjQUY2Vzg). Put `models/` and `data/` folders under the root directory of the project.

## Usage

To train new model

```bash
. train.sh [hs|django]
```

To use trained model for decoding test sets

```bash
. run_trained_model.sh [hs|django]
```

## Dependencies

* Theano
* vprof
* NLTK 3.2.1
* astor 0.6

## Reference

```
@inproceedings{yin17acl,
    title = {A Syntactic Neural Model for General-Purpose Code Generation},
    author = {Pengcheng Yin and Graham Neubig},
    booktitle = {The 55th Annual Meeting of the Association for Computational Linguistics (ACL)},
    address = {Vancouver, Canada},
    month = {July},
    url = {https://arxiv.org/abs/1704.01696},
    year = {2017}
}
```
