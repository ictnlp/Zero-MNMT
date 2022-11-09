# Zero-MNMT 
Source code for the EMNLP 2022 Long findings ["Improving Zero-Shot Multilingual Translation with Universal Representations and Cross-Mappings"](https://arxiv.org/abs/2210.15851).

We propose two training objectives to improve the zero-shot ability of many-to-many multilingual neural machine translation. 

The OT loss is used to bridge the gap between the semantic spaces of different languages, and the AT loss is used to improve the prediction consistency of the MNMT model based on different source languages.  

![](./images/method.png)

## Code  

This code is based on the open source toolkit [fairseq-py](https://github.com/facebookresearch/fairseq).

All the core codes of our method are put in the folder "./zs_nmt".

Codes related to the training objectives is in _label_smoothed_cross_entropy_adapter_zs.py_

_translation_w_langtok.py_ is taken from [mRASP2](https://github.com/PANXiao1994/mRASP2) directly for generation.

## Get Started 

This system has been tested in the following environment.
+ Python version \== 3.7
+ Pytorch version \== 1.7

### Build 
```
pip install --editable ./
```

### Data 
+ IWSLT: https://wit3.fbk.eu/2017-01
+ PC-6: https://github.com/PANXiao1994/mRASP2
+ OPUS-7: https://opus.nlpl.eu/opus-100.php

Following mRASP2, we also append special language tokens "LANG_TOK_XX" at the beginning of both the source and target sentences to indicate the language. Then we mix the multilingual data and train the model with the mixed data. 

### Training
First, pre-train the model only with the cross-entropy loss
```
# path to data_bin
data=
# save dir for the checkpoints
dir=

CUDA_VISIBLE_DEVICES=0,1,2,3  python fairseq_cli/train.py --ddp-backend=no_c10d  $data \
	  --arch transformer_wmt_en_de  --share-all-embeddings  --fp16 --reset-dataloader \
	    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --user-dir zs_nmt  --pre-train \
        --gamma1 0. --gamma2 0. --language-tag-id '3,14,15,18,21' \
	      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
	        --lr 0.0007 --stop-min-lr 1e-09 --dropout 0.3 --seed 9527 \
           --criterion label_smoothed_cross_entropy_zs --label-smoothing 0.1 --weight-decay 0.0\
		    --max-tokens 4096   --save-dir checkpoints/$dir --max-update 20000\
		    --update-freq 2 --no-progress-bar --log-format json --log-interval 25  --save-interval-updates  2000 --keep-interval-updates 40
 ```
+ *--dropout* is set as 0.3 for for IWSLT datasets and 0.1 for PC-6 and OPUS-7 datasets.
+ *--max-update* is set as 20k for IWSLT and 100K for PC-6 and OPUS-7.
+ *--language-tag-id* is the dictionary id (started by 1) of the special language tokens "LANG_TOK_XX". So this should be set dynamically according to the dictionary used for training.

Then fine-tuning the pretrained model:
```
# path to data_bin
data=
# save dir for the checkpoints
dir=
# the pre-trained checkpoint
ckt=

CUDA_VISIBLE_DEVICES=0,1,2,3  python fairseq_cli/train.py --ddp-backend=no_c10d  $data \
	  --arch transformer_wmt_en_de  --share-all-embeddings  --fp16 --reset-dataloader \
	    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --user-dir zs_nmt   --restore-file $ckt \
        --gamma1 0.4 --gamma2 0.002 --language-tag-id '3,14,15,18,21' \
	      --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
	        --lr 0.0007 --stop-min-lr 1e-09 --dropout 0.3 --seed 9527 \
           --criterion label_smoothed_cross_entropy_zs --label-smoothing 0.1 --weight-decay 0.0\
		    --max-tokens 4096   --save-dir checkpoints/$dir --max-update 100000\
		    --update-freq 2 --no-progress-bar --log-format json --log-interval 25  --save-interval-updates  2000 --keep-interval-updates 40
 ```
+ *--max-update* is set as 100k for IWSLT and 300K for PC-6 and OPUS-7.
+ *--gamma1* and *--gamma2* are the hyperparameters of our method. 
