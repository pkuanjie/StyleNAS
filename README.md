# StyleNAS
### Official pytorch implementation of the paper: "Ultrafast Photorealistic Style Transfer via Neural Architecture Search"
####  AAAI 2020 (accepted for oral presentation)

## Make photorealistic style transfer with arbitrary content and style photoes.
With the proposed network architecture named PhotoNAS, you can efficiently produce photorealistic style transfer results of high-resolution content and style inputs.
![](imgs/intro.jpg)

## Accelerate photorealistic style transfer with StyleNAS.
With StyleNAS framework, you can easily accelerate a given style transfer network to a ultrafast one with less parameters at the expense of little decline of style transfer performance.

![](imgs/framework.jpg)


## Environment Requirements
### PhotoNAS
- pytorch 1.3.0
- opencv 3.0

### StyleNAS
- In order to make neural architecture search, StyleNAS is designed to run on a GPU cluster with at least 50 GPU cards and use slurm workload manager. However, you can also run StyleNAS with fewer GPU cards by modifying the code of `style_nas.py` to match your hardware environment.


## Code for PhotoNAS
The proposed PhotoNAS network architecture is in `PhotoNAS` directory of this repository.

###  Train
To train PhotoNAS network for photorealistic style transfer, you should first download the [MS_COCO](http://images.cocodataset.org/zips/unlabeled2017.zip) 2017 unlabelled image and put those images into `training_data` directory. Then run

```
python PhotoNAS/train_decoder.py --training_dataset <absolute-path-of-training-dataset-directory>
```

You may also change other training settings such as epoch number and position to save checkpoints in `PhotoNAS/train_decoder.py`. Please note that `--d_control` specify the network architecture of the trained model. Please keep it intact while training PhotoNAS.

###  Make style transfer
To make style transfer with PhotoNAS, please first train the decoder of PhotoNAS as introduced above or download the trained checkpoint from [here](https://drive.google.com/open?id=15PP0K55jH2tBeWfLAYG7r0LuW0RZvmKd) and put the saved checkpoints into `trained_models_aaai` directory under `PhotoNAS`, then run

```
python PhotoNAS/photo_transfer.py --content <directory-of-input-content-images> --style <directory-of-input-style-images> --save_dir <directory-to-save-produced-images>
```

## Code for StyleNAS
###  Run
To prune a given photorealistic style transfer model, you should first download the training data and put those images into `training_data` directory as introduced above. Then run

```
python style_nas.py
```

This will automatically prune the network architecture. The trained and pruned architectures and its style transfer results will be saved in `configs` directory. The pruning precess and results will be logged into `record.txt` file. You may pick up your favored architecture according to performance/efficiency balance.

## Citation
If you feel this repository is helpful, please cite our paper,
@inproceedings{an2019ultrafast,
  title={Ultrafast Photorealistic Style Transfer via Neural Architecture Search},
  author={An, Jie and Xiong, Haoyi and Huan, Jun and Luo, Jiebo}
  booktitle={AAAI},
  year={2020}
}
