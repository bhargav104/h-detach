#!/usr/bin/env bash
export HOME=`getent passwd kenan | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
 python train.py --id st --caption_model show_tell --input_json da
 ta/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --batch_size 10  --checkpoint_path coco_explore --save_checkpoint_every 6000 --val_images_use 5000 --max_epochs 30 --language_eval 1
 --learning_rate 0.0001
