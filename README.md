# ThiNet-Filter-Selection-Pytorch--ResNet50-Imagenet
This is the Python code for the filter selection of the [ThiNet method](https://arxiv.org/abs/1707.06342). 
<img width="713" alt="Screen Shot 2023-02-27 at 5 00 54 AM" src="https://user-images.githubusercontent.com/40675941/221533223-53e43cd8-22af-43b3-b26f-12c9f33a6c2d.png">.

This code has been designed for ResNet 50 on ImageNet(**[ILSVRC2012](https://www.image-net.org/challenges/LSVRC/2012/)**). This implementation is based on [FPGM](https://github.com/he-y/filter-pruning-geometric-median).

## Requirements:
- PyThon  version : 3.9.12
- PyTorch version : 1.11.0
- TorchVision : 0.12.0

## Explanation:
- `rate_thinet = 0.2` means we prune 20% of the filters in each convolutional layer and keep 80% of the filters.
- `random_data = 10000` means the number of images on the sub-dataset for filter selection by F-ThiNet in 10000.
- `random_entry = 10` means the number of the output instances for calculating the values for filter selection by F-ThiNet in 10.
- In this implementation the pre-trained pytorch weights for [ResNet50](https://pytorch.org/vision/stable/models.html) is used.
- P_task -----> Since filter selection for ThiNet method is time consuming, Filetr selction and fine-tunning must be done seperatly. Select between   
