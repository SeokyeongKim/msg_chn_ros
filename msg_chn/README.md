# msg_chn_wacv20
This repository contains the network configurations (in PyTorch) of our paper "[A Multi-Scale Guided Cascade Hourglass Network for Depth Completion](http://openaccess.thecvf.com/content_WACV_2020/papers/Li_A_Multi-Scale_Guided_Cascade_Hourglass_Network_for_Depth_Completion_WACV_2020_paper.pdf)" by Ang Li, Zejian Yuan, Yonggen Ling, Wanchao Chi, Shenghao Zhang，and Chong Zhang.

## Introduction
Depth completion, a task to estimate the dense depth map from sparse measurement under the guidance from the
high-resolution image, is essential to many computer vision applications. Most previous methods building on fully convolutional networks can not handle diverse patterns in the depth map efficiently and effectively. We propose a multi-scale guided cascade hourglass network to tackle this problem. Structures at different levels are captured by specialized hourglasses in the cascade network with sparse inputs in various sizes. An encoder extracts multiscale features from color image to provide deep guidance
for all the hourglasses. A multi-scale training strategy further activates the effect of cascade stages. With the role of
each sub-module divided explicitly, we can implement components with simple architectures. Extensive experiments show that our lightweight model achieves competitive results compared with state-of-the-art in KITTI depth completion benchmark, with low complexity in run-time.

<p align="center">
  <img src="https://github.com/anglixjtu/msg_chn_wacv20/blob/master/demo/video5.gif" alt="photo not available" height="50%">
</p>


## Network
The implementation of our network is in ```network.py```. It takes the sparse depth and the rgb image (normalized to 0~1) as inputs， outputs the predictions from the last, the second, and the first sub-network in sequence. The output from the last network (```output_d11```) is used for the final test.

    Inputs: input_d, input_rgb
    Outputs: output_d11， output_d12， output_d14
             # outputs from the last, the second, and the first sub-network

※NOTE: We recently modify the architecture by adding the skip connections between the depth encoders and the depth decoders at the previous stage. This vision of network has 32 channels rather than 64 channels in our paper. The 32-channel network performs similarly to the 64-channel network in our paper on the test set, but has a much smaller number of parameters and a shorter run time. You can find more details in [Results](#results)

## Training and Evaluation
- We use an elegant framework written by Abdelrahman Eldesokey to train and evaluate our model. Find more details from [here](https://github.com/abdo-eldesokey/nconv).
- We train our model on KITTI training set, without pre-training on other dataset. To further improve the accuracy，you might consider pre-training with other datasets like Virtual KITTI. 
- Random cropping and left-right-flipping are performed as data augmentation. The training maps are cropped to a
resolution of 1216×352. 
- We adopt a multi-stage scheme during the training process. You can implement the training process as

```python
loss14 = L2Loss(output_d14, label)
loss12 = L2Loss(output_d12, label)
loss11 = L2Loss(output_d11, label)

if epoch < 6:
   loss = loss14 + loss12 + loss11
elif epoch < 11:
   loss = 0.1 * loss14 + 0.1 * loss12 + loss11
else:
   loss = loss11

```
Loss drops very little after 20 epochs. We trained 28 epoches to get the final model.
More training configurations are given in ```params.json```.

### Example usage
#### Training

- For help:

  ```sh
  $ ./train.py --help
  ```

- From scratch (if the weights exist and the images are RGB (uint8), depth (uint16)):

    ```sh
  $ ./train.py  -l /path/to/put/log/
  ```

- From a pretrained model:

  ```sh
  $ ./train.py  -m /path/to/pretrained/ -l /path/to/put/log/
  ```

*Note*: Log will save at workspace/exp_msg_chn/logs/name_given by default if given relative path. To save at any path, use absolute path

**Arguments**

`-exp` (`string`, `default: exp_msg_chn`)

String for workspace if multiple different networks are available. 

`-mode` (`string`, `default: train`)

Either start evaluation only (`eval`) or training the whole dataset (`train`)

`-w`, `--wandb` (`bool`, `default: false`)

Logs everything in the Weights & Biases platform. Defaults at `false`

`-n` (`string`, `default: `)

To name run for weights and biases. If none is given, wandb will create a random name.

`-d`, `--debug` (`bool`, `default: false`)

Verbose for extra information needed to evaluate and debug the code

`-l`, `--log` (`string`, `default: /path/to/workspace/exp_msg_chn/logs/current_time`)

Directory to put logging data


#### To evaluate a network

- It uses the same script as the train but with an extra flag:

  ```sh
  $ ./train.py -mode eval -m /path/to/pretrained
  ```

---

## Results
The performance of our network is given in the table. We validate our model with both the validation dataset (```val```) and the selected depth data (```val_selection_cropped``` ) in KITT dataset. 

|        |  RMSE |  MAE |  iRMSE  | iMAE  | #Params |
|--------|-------|-------|-------|-------|-------|
|validation|821.94|227.94|2.47|0.98|364K|
|selected validation|817.08|224.83|2.48|0.99|364K|
|test|783.49|226.91|2.35|1.01|364K|

You can find our final model and the test results on KITTI data set [here](https://drive.google.com/drive/folders/1botFS752LSEq5NIv0enH2Nh65coil3p5?usp=sharing).

## Citation
If you use our code or method in your work, please cite the following:
```
@inproceedings{li2020multi,
  title={A Multi-Scale Guided Cascade Hourglass Network for Depth Completion},
  author={Li, Ang and Yuan, Zejian and Ling, Yonggen and Chi, Wanchao and Zhang, Chong and others},
  booktitle={The IEEE Winter Conference on Applications of Computer Vision},
  pages={32--40},
  year={2020}
}
```
Please direct any questions to Ang Li at angli522@foxmail.com.

