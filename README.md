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



(4) Compile the `OP` library.
```
cd lib/OP
python setup.py develop
```





## Acknowledgement
This repo is built upon several repos, e.g., [Pointgroup](https://github.com/dvlab-research/PointGroup.gitCloning) [SparseConvNet](https://github.com/facebookresearch/SparseConvNet), [spconv](https://github.com/traveller59/spconv) and [STPLS3D](https://github.com/meidachen/STPLS3D.git). 

## Contact
If you have any questions or suggestions about this repo, please feel free to contact me (lijiang@cse.cuhk.edu.hk).


