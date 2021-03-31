#!/bin/sh
# Distributed under MIT license

gpu_id=0

script_dir=`dirname $0`
main_dir=$script_dir/..
data_dir=$main_dir/data

print_monotonicity() {
    awk '{sum += $2; n++} END {if (n>0) print "LMono:", sum/n;}' ${1}/${2}.mono >> ${1}/eval_log
    awk '{sum += $3; n++} END {if (n>0) print "% Avg Position Increase:", sum/n;}' ${1}/${2}.mono >> ${1}/eval_log
}

evaluate() {
    echo 'Evaluate model' ${2} 'on' ${3} 'set' >> ${1}/${2}/eval_log

    python3 -m sockeye.translate \
    --models ${1}/${2} \
    --beam-size 1 \
    --batch-size 20 \
    --input ${1}/${3}.src \
    --output ${1}/${2}/${3}.out \
    --device-ids ${4} \
    --disable-device-locking

    python3 $script_dir/evaluate.py -o ${1}/${2}/${3}.out -r ${1}/${3}.ref -s ${1}/${3}.src >> ${1}/${2}/eval_log
}

runtransformer() {

    mkdir -p ${1}/${2}/${3}_${4}
    echo 'Train model' ${3}_${4} 'for' ${2} >> ${1}/${2}/${3}_${4}/eval_log
    echo ''

	# train model
    python3 -m sockeye.train \
    --source ${1}/${2}/train.src \
    --target ${1}/${2}/train.trg \
    -vs ${1}/${2}/dev-for-train.src \
    -vt ${1}/${2}/dev-for-train.trg \
    --output ${1}/${2}/${3}_${4} \
    --seed ${4} \
    --batch-type sentence \
    --batch-size 400 \
    --checkpoint-interval 400 \
    --encoder transformer \
    --decoder transformer \
    --num-layers 4:4 \
    --embed-dropout 0.3:0.3 \
    --transformer-model-size 256 \
    --transformer-attention-heads 4 \
    --transformer-feed-forward-num-hidden 512 \
    --transformer-preprocess n \
    --transformer-postprocess dr \
    --transformer-dropout-attention 0.0 \
    --transformer-drophead-attention 0.3 \
    --transformer-dropout-act 0.3 \
    --transformer-dropout-prepost 0.3 \
    --transformer-positional-embedding-type fixed \
    --max-seq-len 85:85 \
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
    --decode-and-evaluate 1000 \
    --device-ids ${5} \
    --disable-device-locking \
    --decode-and-evaluate-device-id ${5} \
    --checkpoint-decoder-beam-size 1 \
    --attention-monotonicity \
    --attention-monotonicity-loss-lambda ${6} \
    --attention-monotonicity-loss-margin ${7} \
    --monotonicity-on-heads ${8}:${9} \
    --monotonicity-on-layers 1:4 \
    --attention-monotonicity-loss-normalize-by-source-length

    # evaluate model on dev set
    evaluate ${1}/${2} ${3}_${4} dev-for-eval ${5}
    echo ''

    # evaluate model on test set
    evaluate ${1}/${2} ${3}_${4} test-for-eval ${5}

	# measure monotonicity
	python3 -m sockeye.score \
    --source ${1}/${2}/test-for-scoring.src \
    --target ${1}/${2}/test-for-scoring.trg \
    --output ${1}/${2}/${3}_${4}/test-for-scoring.mono \
    --model ${1}/${2}/${3}_${4} \
    --batch-type sentence \
    --batch-size 20 \
    --device-ids ${5} \
    --attention-monotonicity-scoring \
    --attention-monotonicity-scoring-margin ${7} \
    --monotonicity-scoring-on-layers 1:4 \
    --monotonicity-scoring-on-heads ${8}:${9} \
    --disable-device-locking

    print_monotonicity ${1}/${2}/${3}_${4} test-for-scoring

}

runrnn() {

    mkdir -p ${1}/${2}/${3}_${4}
    echo 'Train model' ${3}_${4} 'for' ${2} >> ${1}/${2}/${3}_${4}/eval_log
    echo ''

	# train model
    python3 -m sockeye.train \
    --source ${1}/${2}/train.src \
    --target ${1}/${2}/train.trg \
    -vs ${1}/${2}/dev-for-train.src \
    -vt ${1}/${2}/dev-for-train.trg \
    --output ${1}/${2}/${3}_${4} \
    --seed ${4} \
    --batch-type sentence \
    --batch-size 50 \
    --checkpoint-interval 4000 \
    --embed-dropout 0.4:0.4 \
    --encoder rnn \
    --decoder rnn \
    --num-layers 2:1 \
    --max-seq-len 85:85 \
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
    --device-ids ${5} \
    --disable-device-locking \
    --decode-and-evaluate-device-id ${5} \
    --keep-last-params 10 \
    --checkpoint-decoder-beam-size 1 \
    --attention-monotonicity \
    --attention-monotonicity-loss-lambda ${6} \
    --attention-monotonicity-loss-margin ${7} \
    --attention-monotonicity-loss-normalize-by-source-length

    # evaluate model on dev set
    evaluate ${1}/${2} ${3}_${4} dev-for-eval ${5}
    echo ''

    # evaluate model on test set
    evaluate ${1}/${2} ${3}_${4} test-for-eval ${5}

	# measure monotonicity
	python3 -m sockeye.score \
    --source ${1}/${2}/test-for-scoring.src \
    --target ${1}/${2}/test-for-scoring.trg \
    --output ${1}/${2}/${3}_${4}/test-for-scoring.mono \
    --model ${1}/${2}/${3}_${4} \
    --batch-type sentence \
    --batch-size 20 \
    --device-ids ${5} \
    --attention-monotonicity-scoring \
    --attention-monotonicity-scoring-margin ${7} \
    --monotonicity-scoring-on-layers 1:1 \
    --disable-device-locking

    print_monotonicity ${1}/${2}/${3}_${4} test-for-scoring


}


######## TRANSLITERATION EXPERIMENTS ########

for seed in 1 2 3
    do
        for lang in $(cat $main_dir/language-list.txt)
        do
            runtransformer $data_dir $lang base $seed $gpu_id 0.0 0.0 1 4
            runtransformer $data_dir $lang mono_all_heads $seed $gpu_id 0.1 0.0 1 4
            runtransformer $data_dir $lang mono_1_head $seed $gpu_id 0.1 0.0 1 1

            runrnn $data_dir $lang rnn_base $seed $gpu_id 0.0 0.0
            runrnn $data_dir $lang rnn_mono $seed $gpu_id 0.1 0.0

    done
done
