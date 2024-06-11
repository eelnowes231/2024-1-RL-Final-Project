#!/bin/bash
python pretrain_embed.py --epoch 100

python pretrain_embed.py --epoch 100 --modality 'video,audio,text' --fusion early --aggregation concat
python pretrain_embed.py --epoch 100 --modality 'video,audio,text' --fusion early --aggregation mean
python pretrain_embed.py --epoch 100 --modality 'video,audio,text' --fusion late --aggregation concat
python pretrain_embed.py --epoch 100 --modality 'video,audio,text' --fusion late --aggregation mean

python pretrain_embed.py --epoch 100 --modality 'video,audio' --fusion early --aggregation concat
python pretrain_embed.py --epoch 100 --modality 'video,audio' --fusion early --aggregation mean
python pretrain_embed.py --epoch 100 --modality 'video,audio' --fusion late --aggregation concat
python pretrain_embed.py --epoch 100 --modality 'video,audio' --fusion late --aggregation mean