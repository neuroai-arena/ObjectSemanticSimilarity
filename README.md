# Learning Object Semantic Similarity with Self-Supervision

This is the repository used for the paper "Learning Object Semantic Similarity with 
Self-Supervision" published at ICDL 2024. The repository is derived from 
[this repository](https://github.com/trieschlab/SSLTT)

The paper is available [there](https://arxiv.org/pdf/2405.05143)

## Install

```
python3 -m venv ssltt
source ssltt/bin/activate
python3 -m pip install -r requirements.txt
```

## Dataset

This work uses the MVImgNet dataset. The dataloader is based on multiple hdf5 and parquet files built on top on the raw images. We can not provide the exact code for creating hdf5 and parquet files. This aimed to comply with computational constraints. 
Feel free to modify the dataloader accordingly.


Our .parquet contains 5 key columns, which one will also need to reuse key parts of the dataloader:
category,object,frame,length,partition
they denote, respectively, the category id, object instance id, the frame number within the video clip, the length of the clip and the MVImgNet chunk (we used the chunked version).

Feel free to make a pull request with a version that uses directories.

Alternatively, we provide another code for generating the dataset, but note that it was not tested with this code and likely needs modifications:

`python3 scripts/construct_mvi.py --path $DATA_ROOT`

## Train


```

#Standard training
python3 train.py --name normal --data_root $DATASET_LOCATION --contrast time --modules classic,labels,linear_eval --context_json room_assignment_balanced.json`

#only language alignement
python3 train.py --name language_only --data_root $DATASET_LOCATION --contrast classic --modules labels,linear_eval --context_json room_assignment_balanced.json

#only temporal alignment
python3 train.py --name time_only --data_root $DATASET_LOCATION --contrast time --modules classic,linear_eval --context_json room_assignment_balanced.json

#SimCLR
python3 train.py --name simclr --data_root $DATASET_LOCATION --contrast classic --modules classic,linear_eval --context_json room_assignment_balanced.json --jitter 0.8 --grayscale 0.2 --pcrop 1 --min_crop 0.5 --blur 0 --flip 0.5 

#With random assignments of objects to rooms
python3 train.py --name random_context --data_root $DATASET_LOCATION --contrast time --modules classic,labels,linear_eval --context_json room_assignment_balanced_random_0.json 

```

Interesting parameters:

`--crop_number` temporal size of clips

`--p_change_room`: probability to change room when switching objects (p<sub>c</sub>)

`--log_dir`: log directory

## Evaluation

```
python3 train.py --path_load_model save/normal --epoch_load_model 99 --mode odd_one_out --fine_label 1 --fine_proj 0,1,2,3,4,5,6 --context_json room_assignment_balanced.json --data_root $DATASET_LOCATION
```

In odd_one_out{X}.txt: X="1","","3" denotes the odd-one-out accuracy for categories, contexts and objects, respectively.
N in cos{N} denote the layer used for evaluation: "": avgpool, 1: first relu of time projection layer, 2: second relu of time projection layer
3: end of projection layer, 4: first relu of language projection layer, 5: second relu of language projection layer, 6: end of language projection layer.
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details