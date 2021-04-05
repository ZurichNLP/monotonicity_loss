#!/bin/bash

gpu_id=0

script_dir=`dirname $0`
main_dir=$script_dir/..
data_dir=$main_dir/data

nettalk_test=$data_dir/NETtalk.test.src
nettalk_ref=$data_dir/NETtalk.test.trg
cmu_test=$data_dir/CMUDict.test.src
cmu_ref=$data_dir/CMUDict.test.trg


print_monotonicity() {
    awk '{sum += $2; n++} END {if (n>0) print "LMono:", sum/n;}' ${1}/${2}.mono >> ${1}/eval_log
    awk '{sum += $3; n++} END {if (n>0) print "% Avg Position Increase:", sum/n;}' ${1}/${2}.mono >> ${1}/eval_log
}

evaluate() {
    # $1=path-to-dataset, $2=modelname, $3=dev/test set $4=gpu
    echo 'Evaluate model' ${2} 'on' ${3} 'set' >> ${1}/${2}/eval_log

    python3 -m sockeye.translate \
    --models ${1}/${2} \
    --beam-size 1 \
    --batch-size 20 \
    --input ${1}.${3}.src \
    --output ${1}/${2}/${3}.out \
    --device-ids ${4} \
    --skip-topk \
    --length-penalty-alpha 0.0 \
    --disable-device-locking

    echo "acc/PER/ED: " >> ${1}/${2}/eval_log
    python eval.py --prediction ${1}/${2}/${3}.out --reference ${1}.${3}.trg >> ${1}/${2}/eval_log
}

runtransformer() { 
# $1=data_dir, $2=dataset, $3=modelname, $4=seed, $5=gpu, $6=lambda, $7=margin, $8=start-head, $9=end-head, $10=dropout, $11=ff
    mkdir -p ${1}/${2}/${3}_${4}
    echo 'Train model' ${3}_${4} 'for' ${2} >> ${1}/${2}/${3}_${4}/eval_log
    echo ''

	# train model
    python3 -m sockeye.train \
    --source ${1}/${2}.train.src \
    --target ${1}/${2}.train.trg \
    -vs ${1}/${2}.dev.src \
    -vt ${1}/${2}.dev.trg \
    --output ${1}/${2}/${3}_${4} \
    --seed ${4} \
    --batch-type sentence \
    --batch-size 400 \
    --checkpoint-interval 400 \
    --encoder transformer \
    --decoder transformer \
    --num-layers 4:4 \
    --embed-dropout ${10}:${10} \
    --transformer-model-size 256 \
    --transformer-attention-heads 4 \
    --transformer-feed-forward-num-hidden ${11} \
    --transformer-preprocess n \
    --transformer-postprocess dr \
    --transformer-dropout-attention 0.0 \
    --transformer-drophead-attention ${10} \
    --transformer-dropout-act ${10} \
    --transformer-dropout-prepost ${10} \
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
    evaluate ${1}/${2} ${3}_${4} dev ${5}
    echo ''

    # evaluate model on test set
    evaluate ${1}/${2} ${3}_${4} test ${5}

	# measure monotonicity
	python3 -m sockeye.score \
    --source ${1}/${2}.test.src \
    --target ${1}/${2}.test.trg \
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
    
    ## if mono loss on one head, score other heads as well:
    if [[ "$3" == "mono_1_head" ]]; then
        python3 -m sockeye.score \
        --source ${1}/${2}.test.src \
        --target ${1}/${2}.test.trg \
        --output ${1}/${2}/${3}_${4}/test-for-scoring.otherheads.mono \
        --model ${1}/${2}/${3}_${4} \
        --batch-type sentence \
        --batch-size 20 \
        --device-ids ${5} \
        --attention-monotonicity-scoring \
        --attention-monotonicity-scoring-margin ${7} \
        --monotonicity-scoring-on-layers 1:4 \
        --monotonicity-scoring-on-heads 2:4 \
        --disable-device-locking

        echo "Mono scores of heads without Lmono:" >> ${1}/${2}/${3}_${4}/eval_log
        print_monotonicity ${1}/${2}/${3}_${4} test-for-scoring.otherheads
    fi
}

runrnn() {
# $1=data_dir, $2=dataset, $3=modelname, $4=seed, $5=gpu, $6=lambda, $7=margin, 
    mkdir -p ${1}/${2}/${3}_${4}
    echo 'Train model' ${3}_${4} 'for' ${2} >> ${1}/${2}/${3}_${4}/eval_log
    echo ''

	# train model
    python3 -m sockeye.train \
    --source ${1}/${2}.train.src \
    --target ${1}/${2}.train.trg \
    -vs ${1}/${2}.dev.src \
    -vt ${1}/${2}.dev.trg \
    --output ${1}/${2}/${3}_${4} \
    --seed ${4} \
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
    evaluate ${1}/${2} ${3}_${4} dev ${5}
    echo ''

    # evaluate model on test set
    evaluate ${1}/${2} ${3}_${4} test ${5}

	# measure monotonicity
	python3 -m sockeye.score \
    --source ${1}/${2}.test.src \
    --target ${1}/${2}.test.trg \
    --output ${1}/${2}/${3}_${4}/test-for-scoring.mono \
    --model ${1}/${2}/${3}_${4} \
    --batch-type sentence \
    --batch-size 20 \
    --device-ids ${5} \
    --attention-monotonicity-scoring \
    --attention-monotonicity-scoring-margin ${7} \
    --disable-device-locking

    print_monotonicity ${1}/${2}/${3}_${4} test-for-scoring


}


######## TRANSLITERATION EXPERIMENTS ########

for seed in 1 #2 3
    do
        for dataset in CMUDict #NETtalk
        do
            if [[ "$dataset" == "CMUDict" ]]; then
                ff=1024
                dropout=0.2
            elif [[ "$dataset" == "NETtalk" ]]; then
                ff=512
                dropout=0.3
            fi
            # $1=data_dir, $2=dataset, $3=modelname, $4=seed, $5=gpu, $6=lambda, $7=margin, $8=start-head, $9=end-head, $10=dropout, $11=ff
            runtransformer $data_dir $dataset base $seed $gpu_id 0.0 0.0 1 4 $dropout $ff
            runtransformer $data_dir $dataset mono_all_heads $seed $gpu_id 0.1 0.0 1 4 $dropout $ff
            runtransformer $data_dir $dataset mono_1_head $seed $gpu_id 0.1 0.0 1 1 $dropout $ff
 
            runrnn $data_dir $dataset rnn_base $seed $gpu_id 0.0 0.0
            runrnn $data_dir $dataset rnn_mono_m0.0 $seed $gpu_id 0.1 0.0
            runrnn $data_dir $dataset rnn_mono_m0.5 $seed $gpu_id 0.1 0.5
            runrnn $data_dir $dataset rnn_mono_m1.0 $seed $gpu_id 0.1 1.0

    done
done
