#!/bin/sh

working_dir=path-to-dir
data_dir=data
seed=2
model_dir=$working_dir/models/rnn/large/cmu_s${seed}

## Example: lambda=0.1, margin=0.5
modelname=netttalk_rnn_l0.1_m0.5

model=$model_dir/$modelname
train=$data_dir/NETtalk.train
dev=$data_dir/NETtalk.dev

# GPU id
device=0

python3 -m sockeye.train \
--source $train.src \
--target $train.trg \
-vs $dev.src \
-vt $dev.trg \
--output $model \
--seed $seed \
--batch-type sentence \
--batch-size 20 \
--checkpoint-interval 4000 \
--embed-dropout 0.4:0.4 \
--encoder rnn \
--decoder rnn \
--num-layers 2:1 \
--max-seq-len 20:20 \
--label-smoothing 0.0 \
--num-embed 200:200 \
--rnn-cell-type lstm \
--rnn-num-hidden 400 \
--rnn-decoder-hidden-dropout 0.4 \
--word-min-count 1:1 \
--optimizer adam \
--optimized-metric per \
--gradient-clipping-threshold 5 \
--initial-learning-rate 0.001 \
--learning-rate-reduce-num-not-improved 1 \
--learning-rate-reduce-factor 0.5 \
--learning-rate-scheduler-type plateau-reduce \
--learning-rate-warmup 0 \
--max-num-checkpoint-not-improved 7 \
--min-num-epochs 0 \
--max-updates 1001000 \
--weight-init xavier \
--weight-init-scale 3.0 \
--weight-init-xavier-factor-type avg \
--decode-and-evaluate 1000 \
--device-ids $device \
--disable-device-locking \
--decode-and-evaluate-device-id $device \
--attention-monotonicity \
--attention-monotonicity-loss-lambda 0.1 \
--attention-monotonicity-loss-margin 0.5 \
--attention-monotonicity-loss-normalize-by-source-length \
--checkpoint-decoder-beam-size 1
