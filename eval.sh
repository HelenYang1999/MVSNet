#!/usr/bin/env bash
DTU_TESTING="/home/home1/yangqi/dataset/mvsnet/testing_data/dtu/"
CKPT_FILE="./checkpoints/d192/12.10/model_000015.ckpt"
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=/home/home1/yangqi/dataset/mvsnet/testing_data/dtu/ --testlist lists/dtu/test.txt --loadckpt ./checkpoints/d192/12.10/model_000015.ckpt
