# Instance Segmentation for urban scene building



### Requirements
* Python 3.6.0
* Pytorch 1.2.0
* CUDA 10.0

### Virtual Environment
```
conda create -n 3DBuildingInstanceSeg python==3.6
source activate 3DBuildingInstanceSeg
```

### Install 3DBuildingInstanceSeg

(1) Clone from the repository.
```
git clone https://github.com/fullcyxuc/3DBuildingInstanceSeg.git
cd 3DBuildingInstanceSeg
```

(2) Install the dependent libraries.
```
pip install -r requirements.txt
conda install -c bioconda google-sparsehash 
```

(3) For the SparseConv, we apply the implementation of [spconv](https://github.com/traveller59/spconv) as [Pointgroup](https://github.com/dvlab-research/PointGroup.gitCloning) did. The repository is recursively downloaded at step (1). We use the version 1.0 of spconv. 

**Note:** it was modify `spconv\spconv\functional.py` to make `grad_output` contiguous. Make sure you use the modified `spconv`.

* First please download the [spconv](https://github.com/traveller59/spconv), and put it into lib directory

* To compile `spconv`, firstly install the dependent libraries. 
```
conda install libboost
conda install -c daleydeng gcc-5 # need gcc-5.4 for sparseconv
```
Add the `$INCLUDE_PATH$` that contains `boost` in `lib/spconv/CMakeLists.txt`. (Not necessary if it could be found.)
```
include_directories($INCLUDE_PATH$)
```

* Compile the `spconv` library.
```
cd lib/spconv
python setup.py bdist_wheel
```

* Run `cd dist` and use pip to install the generated `.whl` file.



(4) We also use other cuda and cpp extension([pointgroup_ops](https://github.com/dvlab-research/PointGroup/tree/master/lib/pointgroup_ops),[pcdet_ops](https://github.com/yifanzhang713/IA-SSD/tree/main/pcdet/ops)), and put them into the lib, to compile them:
```
cd lib/**  # (** refer to a specific extension)
python setup.py develop
```





## Acknowledgement
This repo is built upon several repos, e.g., [Pointgroup](https://github.com/dvlab-research/PointGroup.gitCloning) [SparseConvNet](https://github.com/facebookresearch/SparseConvNet), [spconv](https://github.com/traveller59/spconv), [IA-SSD](https://github.com/yifanzhang713/IA-SSD/tree/main/pcdet/ops) and [STPLS3D](https://github.com/meidachen/STPLS3D.git). 

