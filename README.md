# Requirements

PyTorch >= 1.00

Python >= 3.6

Tensorflow + TensorboardX


# Prepare

```
python preprocess.py \
    --source-lang en \
    --target-lang zh \
    --workers 32 \
    --trainpref ~/data/medical_mt_extended/tigermed_en32k_zh32k/train_sp \
    --validpref ~/data/medical_mt_extended/tigermed_en32k_zh32k/dev_sp \
    --testpref ~/data/medical_mt_extended/tigermed_en32k_zh32k/test_sp \
    --destdir ~/data/medical_mt_extended/tigermed_en32k_zh32k/tigermed_en32k_zh32k_bin
```

# Train

4GPU-Medical:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ~/data/medical_mt_extended/tigermed_en32k_zh32k/tigermed_en32k_zh32k_bin \
    --arch transformer_wmt_en_de \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0007 \
    --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --weight-decay 0.0 \
    --max-tokens 4096 \
    --update-freq 2 \
    --no-progress-bar \
    --log-format json \
    --log-interval 50 \
    --save-interval-updates 1000 \
    --keep-interval-updates 20 \
    --max-epoch 200 \
    --fp16 \
    --ddp-backend no_c10d \
    --save-dir ~/model/medical_mt/fairseq_enzh_baseline/checkpoints \
    --tensorboard-logdir ~/model/medical_mt/fairseq_enzh_baseline/wmt_ende_conf_4gpu_tb_log
```

2GPU-Medical:

```
CUDA_VISIBLE_DEVICES=0,1 python train.py ~/data/medical_mt_extended/tigermed_en32k_zh32k/tigermed_en32k_zh32k_bin \
    --arch transformer_wmt_en_de \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0007 \
    --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --weight-decay 0.0 \
    --max-tokens 3328 \
    --update-freq 4 \
    --no-progress-bar \
    --log-format json \
    --log-interval 50 \
    --save-interval-updates 1000 \
    --keep-interval-updates 20 \
    --max-epoch 200 \
    --fp16 \
    --ddp-backend no_c10d \
    --save-dir ~/model/medical_mt/fairseq_enzh_baseline_2gpu_3328/checkpoints \
    --tensorboard-logdir ~/model/medical_mt/fairseq_enzh_baseline_2gpu_3328/wmt_ende_conf_2gpu_tb_log
```

4GPU-WMT

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py ~/data/wmt17_medical_segmented_by_wmt17_sp_model/wmt_bin \
    --arch transformer_wmt_en_de \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --lr 0.0007 \
    --min-lr 1e-09 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --weight-decay 0.0 \
    --max-tokens 4000 \
    --update-freq 2 \
    --no-progress-bar \
    --log-format json \
    --log-interval 50 \
    --save-interval-updates 1000 \
    --keep-interval-updates 20 \
    --fp16 \
    --ddp-backend no_c10d \
    --save-dir ~/model/medical_mt/fairseq_wmt_enzh/checkpoints \
    --tensorboard-logdir ~/model/medical_mt/fairseq_wmt_enzh/wmt_enzh_4gpu_tb_log

```

```
tensorboard --logdir=~/model/medical_mt/fairseq_enzh_baseline/wmt_ende_conf_4gpu_tb_log
```


# Average Checkpoints (Optional)

```
PYTHONPATH=~/fairseq-dev/ python ~/fairseq-dev/scripts/average_checkpoints.py \
    --inputs checkpoint200.pt checkpoint199.pt checkpoint198.pt checkpoint197.pt checkpoint196.pt \
    --output averaged_196-200.pt
```


# Score

Calculate the **normalized** BLEU score on the test (`--gen-subset valid` for dev) set.

```
CUDA_VISIBLE_DEVICES=0 python generate.py ~/data/medical_mt_extended/tigermed_en32k_zh32k/tigermed_en32k_zh32k_bin \
	--gen-subset test \
    --remove-bpe sentencepiece \
    --sacrebleu-zh \
    --beam 4 \
    --batch-size 64 \
    --lenpen 0.6 \
    --quiet \
    --rename \
    --path ~/model/medical_mt/fairseq_enzh_baseline/checkpoints/averaged_196-200.pt
```


# Interactive

Translate sentences from given input to given output.

```
CUDA_VISIBLE_DEVICES=0 python interactive.py ~/data/medical_mt_extended/tigermed_en32k_zh32k/tigermed_en32k_zh32k_bin \
    --path ~/model/medical_mt/fairseq_enzh_baseline/checkpoints/averaged_196-200.pt \
    --remove-bpe sentencepiece \
    --beam 4 \
    --lenpen 0.6 \
    --input ~/data/medical_mt_extended/tigermed_en32k_zh32k/test_sp.en \
    --batch-size 64 \
    --buffer-size 64 \
    --output ~/data/medical_mt_extended/tigermed_en32k_zh32k/test_test.zh
```

You can then manually calculate the BLEU score.

```
cat test_test.zh | sacrebleu test_raw.zh --tok zh -b
```

Note: `test_raw.zh` has not been normalized, so please unnormalized `test_test.zh` if you want to calculate the **standard** BLEU score.

