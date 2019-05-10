# Requirements

PyTorch >= 1.00

Python >= 3.6

Tensorflow + TensorboardX



# Prepare

```
python preprocess.py \
    --source-lang en \
    --target-lang zh \
    --trainpref ~/data/medical_mt_extended/tigermed_en32k_zh32k/train_sp \
    --validpref ~/data/medical_mt_extended/tigermed_en32k_zh32k/dev_sp \
    --testpref ~/data/medical_mt_extended/tigermed_en32k_zh32k/test_sp \
    --destdir ~/data/medical_mt_extended/tigermed_en32k_zh32k/tigermed_en32k_zh32k_bin
```



# Train

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
    --save-dir ~/model/medical_mt/fairseq_enzh_baseline/checkpoints \
    --update-freq 2 \
    --no-progress-bar \
    --log-format json \
    --log-interval 50 \
    --save-interval-updates 1000 \
    --keep-interval-updates 20 \
    --tensorboard-logdir ~/model/medical_mt/fairseq_enzh_baseline/wmt_ende_conf_4gpu_tb_log
```

```
tensorboard --logdir=~/model/medical_mt/fairseq_enzh_baseline/wmt_ende_conf_4gpu_tb_log
```



# Score

Calculate the **normalized** BLEU score on the test (`--gen-subset valid` for dev) set.

```
CUDA_VISIBLE_DEVICES=0 python generate.py ~/data/medical_mt_extended/tigermed_en32k_zh32k/tigermed_en32k_zh32k_bin \
	--gen-subset test \
    --path ~/model/medical_mt/fairseq_enzh_baseline/checkpoints/checkpoint_best.pt \
    --remove-bpe sentencepiece \
    --sacrebleu-zh \
    --beam 4 \
    --batch-size 64 \
    --lenpen 0.6 \
    --quiet
```



# Interactive

Translate sentences from given input to given output.

```
CUDA_VISIBLE_DEVICES=0 python interactive.py ~/data/medical_mt_extended/tigermed_en32k_zh32k/tigermed_en32k_zh32k_bin \
    --path ~/model/medical_mt/fairseq_enzh_baseline/checkpoints/checkpoint_best.pt \
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

