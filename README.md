# Code  Running Instructions

## requirement  
pytorch >= 0.4 
visdom  
CUDA >= 8.0  

## dataset  
PA-100K dataset  

|--Hydraplus  
|----data  
|------PA-100K  
|--------annotation  
|--------ralease_data  
|----------release_data  

## 1. train

### 1.1 train M-net fundamental checkpoint
As the paper said:
> We train the HP-net in a stage-wise fashion. Initially,a plain M-net is trained to learn the fundamental pedestrian features.
- We can get fundamental pedestrian features and save the checkpoint via :
```
python train.py -m  MNet
```

> Then the M-net is duplicated three times to construct the AF-net with adjacent MDA modules

- To get AF-net checkpoint:
```
python train.py -m AF1 -mpath MNet/checkpoint_epoch_60
python train.py -m AF2 -mpath MNet/checkpoint_epoch_60
python train.py -m AF3 -mpath MNet/checkpoint_epoch_60
```


### 1.2 fine-tuning existing checkpoint  

> Since each MDA module consists of three branches where the attention map masks adjacent inception blocks, thus in each branch we only fine-tune the blocks after the attention-operated block.

    python train.py -m AF1 -p AF1/checkpoint_epoch_0  
    python train.py -m AF2 -p AF2/checkpoint_epoch_0  
    python train.py -m AF3 -p AF3/checkpoint_epoch_0  

### 1.3 train the remaining GAP and FC layers

> After separately fine-tuning three MDA modules in AF-net, we fix both the M-net and AF-net and train the remaining GAP and FC layers.

```
python train.py -m HP  -mpath  MNet/checkpoint_epoch_60 -af1path AF1/checkpoint_epoch_0 -af2path AF2/checkpoint_epoch_0 -af3path AF3/checkpoint_epoch_0 
```


## 2.test  
    python test.py -m {AF1|AF2|AF3|HP|MNet} -p checkpointpath  
#### example:  
    python test.py -m AF1 -p AF1/checkpoint_epoch_0
    python test.py -m AF2 -p AF2/checkpoint_epoch_0
    python test.py -m AF3 -p AF3/checkpoint_epoch_0
    python test.py -m MNet -p MNet/checkpoint_epoch_0
    python test.py -m HP -p HP/checkpoint_epoch_0
## 3.show  
    python show.py -m {AF1|AF2|AF3|HP|MNet} -p checkpointpath  
#### example:  
    python show.py -m AF1 -p AF1/checkpoint_epoch_0
    python show.py -m AF2 -p AF2/checkpoint_epoch_0
    python show.py -m AF3 -p AF3/checkpoint_epoch_0
    python show.py -m MNet -p MNet/checkpoint_epoch_0
    python show.py -m HP -p HP/checkpoint_epoch_0

![](/home/tay/HydraPlus-Net/Hydraplus_Net/img/show.png)