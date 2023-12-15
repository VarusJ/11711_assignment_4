# Assignment 4 for ANLP 11-711 -- Group 41
The repo is initially forked from our assignment 3 repo. We resue some training code from assignment 3, and add code for data augmentation as well as transfer learning.

Usage:
- `pip install -r requirements.txt`
- run `accelerate config` to configure your accelerate settings
- For transfer learning, run `accelerate launch {target_file}.py`
- To replicate our augmented datasets, cd to `11711_assignment_4/code/data_aug` folder, choose an augmentation method and run the corresponding notebook step by step
