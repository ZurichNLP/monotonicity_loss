#!/bin/sh

working_dir=path-to-dir
data_dir=data
nettalk_test=$data_dir/NETtalk.test.src
nettalk_ref=$data_dir/NETtalk.test.trg

model_dir=path-to-models
eval_dir=path-to-save-eval-file

## Example: evaluate/score RNN with margin = 1.0 (trained with lambda=0.1, lambda is irrelevant for inference/scoring though)
modelname=netttalk_rnn_l1.0_m1.0
model=$model_dir/$modelname
# GPU id
device=0
margin=1.0

## inference: PER and ACC
python3 -m sockeye.translate \
--device-ids $device \
--model $model \
-i $nettalk_test \
-o decoded/$modelname.predictions \
--batch-size 20 \
--beam-size 1 \
--skip-topk \
--length-penalty-alpha 0 \
--disable-device-locking

# phoneme-error-rate and accuracy
echo "acc/PER/ED: " >> $eval_dir/$modelname.eval
python eval.py --prediction decoded/$modelname.predictions --reference $nettalk_ref >> $eval_dir/$modelname.eval

###########
# SCORING #
###########

## scoring, example RNN:
python3 -m sockeye.score \
--source $nettalk_test \
--target $nettalk_ref \
--model $model \
--output monotonicity_scores/$modelname.scores \
--batch-type sentence \
--batch-size 20 \
--device-ids $device \
--attention-monotonicity-scoring \
--attention-monotonicity-scoring-margin $margin \
--disable-device-locking

# get avg monotonicity score (=loss) and percentage of target positions with an increased average attention
# score.py will print 3 scores: negative log probabilities, monotonicity score, average percentage of target positions with increased attention
echo "avg monotonicity score with margin $margin: " >> $eval_dir/$modelname.eval
awk '{ sum += $2; n++ } END { if (n > 0) print sum / n; }' monotonicity_scores/$modelname.scores >> $eval_dir/$modelname.eval
echo "percentage of target positions with increased attention with margin $margin"  >> $eval_dir/$modelname.eval
awk '{ sum += $3; n++ } END { if (n > 0) print sum / n; }' monotonicity_scores/$modelname.scores >> $eval_dir/$modelname.eval


## scoring, Transformer 
# Example with margin=0, score all heads, all layers (for a model with 4 heads/layers):
python3 -m sockeye.score \
--source $nettalk_test \
--target $nettalk_ref \
--model $model \
--output monotonicity_scores/$modelname.scores \
--batch-type sentence \
--batch-size 20 \
--device-ids $device \
--attention-monotonicity-scoring \
--attention-monotonicity-scoring-margin $margin \
--monotonicity-scoring-on-heads 1:4 \
--monotonicity-scoring-on-layers 1:4 \
--disable-device-locking

echo "avg monotonicity score with margin $margin: " >> $eval_dir/$modelname.eval
awk '{ sum += $2; n++ } END { if (n > 0) print sum / n; }' monotonicity_scores/$modelname.scores >> $eval_dir/$modelname.eval
echo "percentage of target positions with increased attention with margin $margin, all heads, all layers"  >> $eval_dir/$modelname.eval
awk '{ sum += $3; n++ } END { if (n > 0) print sum / n; }' monotonicity_scores/$modelname.scores >> $eval_dir/$modelname.eval

## scoring, Transformer example with margin=0, score only first head, all layers (for a model with 4 heads/layers):
python3 -m sockeye.score \
--source $nettalk_test \
--target $nettalk_ref \
--model $model \
--output monotonicity_scores/$modelname.scores \
--batch-type sentence \
--batch-size 20 \
--device-ids $device \
--attention-monotonicity-scoring \
--attention-monotonicity-scoring-margin $margin \
--monotonicity-scoring-on-heads 1:1 \
--monotonicity-scoring-on-layers 1:4 \
--disable-device-locking

echo "avg monotonicity score with margin $margin: " >> $eval_dir/$modelname.eval
awk '{ sum += $2; n++ } END { if (n > 0) print sum / n; }' monotonicity_scores/$modelname.scores >> $eval_dir/$modelname.eval
echo "percentage of target positions with increased attention with margin $margin, first head, all layers"  >> $eval_dir/$modelname.eval
awk '{ sum += $3; n++ } END { if (n > 0) print sum / n; }' monotonicity_scores/$modelname.scores >> $eval_dir/$modelname.eval

###########################
# Attention Visualization #
###########################
outdir=attention_visualizations/nettalk/$modelname
mkdir -p $outdir

# Example: visualize attention for heads 1 and 3 in layers 1 and 4 for a transformer model with 4 heads/layers
# Default: uses params.best, use --checkpoint to indicate a specific checkpoint to visualize
# Note: batch size needs to be > 1
# Visualization examples in attention_visualizations (score below target = monotonicity loss for this sample - in the examples, this is 0 for both samples, i.e. attention is monotonic if measured with a margin of 0)

for layer in 1 4
do
    for head in 1 3
    do
        python3 -m sockeye.score \
        --source 1sample.in \
        --target 1sample.out \
        --model $model \
        --batch-type sentence \
        --batch-size 1 \
        --device-ids $device \
        --attention-monotonicity-scoring \
        --attention-monotonicity-scoring-margin $margin \
        --monotonicity-scoring-on-heads $head:$head \
        --monotonicity-scoring-on-layers $layer:$layer \
        --disable-device-locking \
        --print-attention-scores \
        --output attention_visualizations/$modelname
    done 
done
