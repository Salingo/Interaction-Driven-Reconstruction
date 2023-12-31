#! /usr/bin/env bash

git clone https://github.com/guochengqian/openpoints.git

cd openpoints/cpp/pointnet2_batch
python setup.py install

cd ../subsampling
python setup.py build_ext --inplace

cd ../pointops
python setup.py install

cd ../chamfer_dist
python setup.py install --user

cd ../emd
python setup.py install --user

cd ../../..
