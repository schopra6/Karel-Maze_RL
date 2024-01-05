
# RL-MAZE

![alt text](https://github.com/schopra6/Karel-Maze_RL/blob/main/images/karel.png?raw=true)

## Problem Statement 

To solve a given Maze, one has to use a sequence of commands that transforms the Pre-grid to Post-grid. The objective is to use small number of commands to solve the task. Figure 1a shows a Karel task with a solution in Figure 1b that uses minimal-sized command sequence.

**AVATAR**: A Karel AVATAR is characterized by its current location and orientation. Its orientation
can be one of {NORTH, EAST SOUTH, WEST} and its location can be any of the grid-cells. The
blue dart in Figure 1a depicts the location and orientation of the AVATAR. The AVATAR can move
around the grid and is directed via different Karel commands as will be described below.

Description of Karel commands <br />
• **move**: This moves the AVATAR one grid-cell in the direction it is currently oriented in. If the
AVATAR hits a WALL or the grid-boundary, then the AVATAR “crashes” and the program terminates. <br />
• **turnLef**t: This orients the AVATAR in the direction left of its current orientation.<br />
• **turnRight**: This orients the AVATAR in the direction right of its current orientation.<br />
• **pickMarker**: This removes a MARKER from the current location (grid-cell) of the AVATAR, if present. If no MARKER is present, the AVATAR “crashes” and the program terminates. <br />
• **putMarker**: This adds a MARKER on the current location (grid-cell) of the AVATAR, if no MARKER is present. If a MARKER is already present, the AVATAR “crashes” and the program terminates. <br />
• **finish**: This command is sent to the AVATAR to indicate end of a sequence of commands.<br />
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


