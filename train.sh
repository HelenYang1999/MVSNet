#!/usr/bin/env bash
MVS_TRAINING="/home/yangqi/dataset/mvsnet/training_data/dtu_training/"
python train.py --dataset=dtu_yao --batch_size=4 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints/d192 $@
python train.py --dataset=dtu_yao --batch_size=2 --trainpath=/home/home1/yangqi/dataset/mvsnet/training_data/dtu/ --trainlist=lists/dtu/train.txt --testlist=lists/dtu/test.txt --numdepth=192 --logdir=./checkpoints/d192
python train.py --dataset=dtu_yao --batch_size=1 --trainpath=/home/yangqi/dataset/mvsnet/training_data/dtu/ --trainlist=lists/dtu/train.txt --testlist=lists/dtu/test.txt --numdepth=192 --logdir=./checkpoints/d192
