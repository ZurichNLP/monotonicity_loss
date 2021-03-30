#!/bin/sh

working_dir=path-to-dir
data_dir=data
seed=2
model_dir=$working_dir/models/transformer/cmu_s${seed}

## Example: lambda=0.1, margin=0.0, loss only on first head, on all layers
modelname=cmu_trs_l1.0_1h_all

model=$model_dir/$modelname
train=$data_dir/CMUDict.train
dev=$data_dir/CMUDict.dev

# GPU id
device=0
dropout=0.2
seed=2
ff=1024

python3 -m sockeye.train \
--source $train.src \
--target $train.trg \
-vs $dev.src \
-vt $dev.trg \
--output $model \
--seed $seed \
--batch-type sentence \
--batch-size 400 \
--checkpoint-interval 400 \
--embed-dropout $dropout:$dropout \
--encoder transformer \
--decoder transformer \
--num-layers 4:4 \
--transformer-model-size 256 \
--transformer-attention-heads 4 \
--transformer-feed-forward-num-hidden $ff \
--transformer-preprocess n \
--transformer-postprocess dr \
--transformer-dropout-attention 0.0 \
--transformer-drophead-attention $dropout \
--transformer-dropout-act $dropout \
--transformer-dropout-prepost $dropout \
--transformer-positional-embedding-type fixed \
--max-seq-len 20:20 \
--label-smoothing 0.1 \
--weight-tying \
--weight-tying-type trg_softmax \
--num-embed 256:256 \
--word-min-count 1:1 \
--optimizer adam \
--optimizer-params beta2:0.98 \
--optimized-metric per \
--gradient-clipping-threshold -1 \
--initial-learning-rate 0.001 \
--learning-rate-scheduler-type fixed-rate-inv-sqrt-t \
--learning-rate-warmup 4000 \
--max-num-checkpoint-not-improved 10 \
--min-num-epochs 0 \
--max-updates 1001000 \
--weight-init xavier \
--weight-init-scale 3.0 \
--weight-init-xavier-factor-type avg \
--decode-and-evaluate 5600 \
--device-ids $device \
--disable-device-locking \
--decode-and-evaluate-device-id $device \
--attention-monotonicity \
--attention-monotonicity-loss-lambda 1.0 \
--attention-monotonicity-loss-margin 0.0 \
--monotonicity-on-heads 1:1 \
--monotonicity-on-layers 1:4 \
--attention-monotonicity-loss-normalize-by-source-length \
--checkpoint-decoder-beam-size 1
