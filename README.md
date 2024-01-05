# README
## Directory Structure
``` bash
├── datasets
│   ├── data
│   ├── data_easy
│   └── data_medium
├── environment_imitation.py
├── environment.py
├── evaluate.py
├── generate_seq.py
├── README.md
├── requirements.txt
├── train_imitation_learning.py
├── models
└── train.py


```


### download the required packages
`pip install -r requirements.txt`

How to run the code?
===========



[1] To evaluate the results


`python evaluate.py --model_type bc --datapath datasets/data/val/task --model_path models/bc_trainer`


[2] To generate the sequence

`python generate_seq.py --model_type bc --datapath test_without_seq/task/ --model_path models/bc_trainer`

```
usage: generate_seq.py  [--model_type] [--datapath] [--model_path]


positional arguments:
  --model_type          type of model to train on(bc)
  --datapath            Path of the data file in a given format
  --model_path          pretrained model path with model name
```

Generates a sequence in the same directory under seq folder

Generated sequence are available in test_without_seq/seq/
  
## Versions

Python 3.8.8


