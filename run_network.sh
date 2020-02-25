#!/bin/sh
cd ../..
python setup.py install
cd samples/endovis
python train.py

