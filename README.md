# SSA-GAN: End-to-End Time-Lapse Video Generation with Spatial Self-Attention


## Environment
+ Python >= 3.6
+ PyTorch >= 0.4.1

## Usage.
1. Download the cloud dataset [here](https://drive.google.com/file/d/1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo/view).
2. Please set the above path to `--data_train` and `--data_test` in the `src/config/parameter.py`
3. Run `bash scripts/train_stage1.sh`

## Generated samples
+ Please see the `./samples/*.mp4`
+ More experimental results including generated videos can be seen [here](http://mm.cs.uec.ac.jp/horita-d/ssagan/).

## Citing
+ Our paper.
```
@InProceedings{Horita_2019_ACPR,
    author = {Daichi, Horita and Keiji, Yanai},
    title = {SSA-GAN: End-to-End Time-Lapse Video Generation with Spatial Self-Attention},
    booktitle = {The Asian Conference on Pattern Recognition (ACPR)},
    year = {2019}
}
```

+ The paper which proposes the cloud dataset.
```
@InProceedings{Xiong_2018_CVPR,
    author = {Xiong, Wei and Luo, Wenhan and Ma, Lin and Liu, Wei and Luo, Jiebo},
    title = {Learning to Generate Time-Lapse Videos Using Multi-Stage Dynamic
    Generative Adversarial Networks},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition
    (CVPR)},
    month = {June},
    year = {2018}
}
```