# Adversarial Attack against 3D Point Cloud Classifier

This repository is the code release of our AAAI'20 paper,
"Robust Adversarial Objects against Deep Learning Models",
which can be found [here](http://jin.ece.ufl.edu/papers/AAAI2020_PointNet_CR.pdf).

## Introduction

In this work, we proposed an adversarial attack against PointNet++, a 3D point
cloud classifier, while considering creating 3D physical objects. We tried to
make sure that the point clouds we generate can produce watertight objects,
which can be used for 3D printing. Also we evaluated the adversarial examples with
several existing defense mechanisms for 3D data introduced in previous work.
For more information, please refer to our paper.

## Installation

The code is tested under Python 3.7 with Tensorflow 1.13.2 in Ubuntu 18.04.

### Prerequisites

- Tensorflow < 2.0.0. Usually you can install it by one of the following command:
```
$ pip3 install tensorflow-gpu<2.0.0
$ conda install tensorflow-gpu<2.0.0   # (if you use Anaconda)
```
Or with specific version:
```
$ pip3 install tensorflow-gpu==1.13.2
$ conda install tensorflow-gpu==1.13.2   # (if you use Anaconda)
```
We recommend installing the one with GPU support for better performance.

- trimesh (used for re-sampling points from meshes)
```
$ pip3 install trimesh
```

- Other well-known stuffs such as numpy, scipy, scikit-learn, tqdm. Some of them will be installed
  automatically as dependencies when you install Tensorflow.
```
$ pip3 install numpy scipy scikit-learn tqdm
```

### Compile TF Operators Introduced by PointNet++

Navigate to `tf_ops` directory and run `compile_all.sh`.
```
$ cd tf_ops && sh compile_all.sh
```

If there are files/executables that can not be found correctly, you might have to modify the
path in the script (such as the path to nvcc, which usually
comes with the NVIDIA Cuda installation).

## Usage

We used the modified dataset ([ModelNet40](https://modelnet.cs.princeton.edu/)) 
which are re-sampled by the PointNet++ authors, which can
be downloaded [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip).
For more information, see [here](https://github.com/charlesq34/pointnet2#shape-classification).
Move the uncompressed data folder to `$PROJECT_ROOT/data/modelnet40_normal_resampled`.

### Training and Evaluation
We have already provided trained models used in our paper in `graph/` directory, which are
PointNet++ with Single Scale Grouping (SSG). If you want to train your own model, please refer to
[PointNet++ repository](https://github.com/charlesq34/pointnet2#usage) for more information.

### Attack

Most of the code for attacking can be found in `attack.py`. To run the attack,
you need to choose the attack target, the input point cloud, and then simply type:
```
$ python3 attack.py --model ./graph/ssg-1024.pb --target <TARGET> --outputdir <OUTPUT> <PATH_TO_INPUT_FILE>
```

For example,
```
$ python3 attack.py --model ./graph/ssg-1024.pb --target flower_pot --outputdir outputs ./data/modelnet40_normal_resampled/bottle/bottle_0001.txt
```

For more information, run `python3 attack.py --help`.

### Inference

Most of the inference code is written in `inference.py`. To inference the model,
run:
```
$ python3 inference.py --model ./graph/ssg-1024.pb --rounds <TIMES_TO_INFERENCE> <PATH_TO_INPUT_FILE>
```

For example,
```
$ python3 inference.py --model ./graph/ssg-1024.pb --rounds 16 ./data/modelnet40_normal_resampled/bottle/bottle_0001.txt
```

For more information, run `python3 inference.py --help`.

### Defenses

#### Random Point Selection
Randomly pick a subset of points from input point clouds for inference. Add `--rr` to the inference command.
For example,
```
$ python3 inference.py --model ./graph/ssg-1024.pb --rounds 16 --rr ./data/modelnet40_normal_resampled/bottle/bottle_0001.txt
```

#### kNN Outlier Removal
Remove outlier points using kNN distance. Add `--knn` to the inference command with two arguments: `k` and `alpha`.
For example,
```
$ python3 inference.py --model ./graph/ssg-1024.pb --rounds 16 --rr --knn 5 0.1 ./data/modelnet40_normal_resampled/bottle/bottle_0001.txt
```

#### Gaussian Noise
For this defense, the threshold are calculated using the original data (given in the ModelNet40 dataset).
The threshold can be found automatically by the following command:
```
$ python3 statistic_defense.py --model ./graph/ssg-1024.pb --stat <SIGMA_SQUARE> --clean ./data/modelnet40_normal_resampled/ <PATH_TO_INPUT_FILE>
```

For example,
```
$ python3 statistic_defense.py --model ./graph/ssg-1024.pb --stat 0.001 --clean ./data/modelnet40_normal_resampled/ ./data/modelnet40_normal_resampled/bottle/bottle_0001.txt
```

Or you can specify the threshold directly:
```
$ python3 statistic_defense.py --model ./graph/ssg-1024.pb --stat 0.001 --threshold 0.005 ./data/modelnet40_normal_resampled/ ./data/modelnet40_normal_resampled/bottle/bottle_0001.txt
```

For more information, run `python3 statistic_defense.py --help`.

## Surface Reconstruction and Point Resampling
We use the [Screened Poisson Surface Reconstruction](http://www.cs.jhu.edu/~misha/MyPapers/ToG13.pdf),
which is introduced by Michael Kazhdan and Hugues Hoppe, and we choose the implementation provided by
the MeshLab software. The official website can be found [here](http://www.meshlab.net/).
The reconstruction functionality can be found at `Filters > Remeshing, Simplification and Reconstruction > Screened Surface Reconstruction`
from the menu bar at the top of the main window. Or you can try the one released by the author of the reconstruction algorithm:
[https://github.com/mkazhdan/PoissonRecon](https://github.com/mkazhdan/PoissonRecon).

The resampling is done using the Trimesh package with the following simple code in Python:

For closest sampling described in paper:
```python
import meshio
import trimesh

def closest_sample():
    adv_pc = meshio.loadmesh('/path/to/adversarial/point/cloud')
    recon_mesh = trimesh.load('/path/to/reconstructed/mesh', file_type='ply')  # Change file_type if necessary

    closest, _, _ = trimesh.proximity.closest_point(recon_mesh, adv_pc)
    return closest
```

For random sampling:
```python
import trimesh

def random_sample():
    recon_mesh = trimesh.load('/path/to/reconstructed/mesh', file_type='ply')  # Change file_type if necessary

    randomly, _ = trimesh.sample.sample_surface(recon_mesh, count=10000)
    return randomly
```

## Acknowledgement

This work is based on [PointNet++](https://github.com/charlesq34/pointnet2),
which is introduced by
<a href="http://charlesrqi.com" target="_blank">Charles R. Qi</a>,
<a href="http://stanford.edu/~ericyi">Li (Eric) Yi</a>,
<a href="http://ai.stanford.edu/~haosu/" target="_blank">Hao Su</a>,
<a href="http://geometry.stanford.edu/member/guibas/" target="_blank">Leonidas J. Guibas</a>
from Stanford University.
