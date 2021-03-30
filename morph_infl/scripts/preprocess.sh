#!/bin/sh
# Distributed under MIT license

# This script preprocesses CoNLL-SIGMORPHON 2017 shared task datasets.

script_dir=`dirname $0`
main_dir=$script_dir/..
high_dir=$main_dir/high
medium_dir=$main_dir/medium

mkdir -p $high_dir
mkdir -p $medium_dir

# Download the original data
git clone https://github.com/sigmorphon/conll2017 $main_dir/orig_data

# Create a list of all the languages
for train_file in $main_dir/orig_data/all/task1/*train-high; do no_prefix=${train_file##*/} ; echo ${no_prefix%-train-high} ; done > $main_dir/language-list.txt

# Preprocess the training data for the high-resource settings
for lang in $(cat $main_dir/language-list.txt)
do
    file=$main_dir/orig_data/all/task1/$lang-train-high
    awk -F '\t' '{print $3, "<sep>"}' $file | sed 's/;/ /g' > $file.tags
    awk -F '\t' '{print $1}' $file | sed 's/ /⌀/g' | sed 's/./& /g' | sed 's/ $//g'  > $file.lemma
    paste -d ' ' $file.tags $file.lemma > $high_dir/$lang-train.src
    awk -F '\t' '{print $2}' $file | sed 's/ /⌀/g' | sed 's/./& /g' | sed 's/ $//g' > $high_dir/$lang-train.trg
done

# Preprocess the training data for the medium-resource settings
for lang in $(cat $main_dir/language-list.txt)
do
    file=$main_dir/orig_data/all/task1/$lang-train-medium
    awk -F '\t' '{print $3, "<sep>"}' $file | sed 's/;/ /g' > $file.tags
    awk -F '\t' '{print $1}' $file | sed 's/ /⌀/g' | sed 's/./& /g' | sed 's/ $//g'  > $file.lemma
    paste -d ' ' $file.tags $file.lemma > $medium_dir/$lang-train.src
    awk -F '\t' '{print $2}' $file | sed 's/ /⌀/g' | sed 's/./& /g' | sed 's/ $//g' > $medium_dir/$lang-train.trg
done

# Preprocess the development data (identical for both settings), save reference for evaluation
for lang in $(cat $main_dir/language-list.txt)
do
    file=$main_dir/orig_data/all/task1/$lang-dev
    awk -F '\t' '{print $3, "<sep>"}' $file | sed 's/;/ /g' > $file.tags
    awk -F '\t' '{print $1}' $file | sed 's/ /⌀/g' | sed 's/./& /g' | sed 's/ $//g'  > $file.lemma
    paste -d ' ' $file.tags $file.lemma > $high_dir/$lang-dev.src
    awk -F '\t' '{print $2}' $file | sed 's/ /⌀/g' | sed 's/./& /g' | sed 's/ $//g' > $high_dir/$lang-dev.trg
    cp $file $high_dir/$lang-dev.ref
    awk -F '\t' '{print $1}' $file > $high_dir/$lang-dev.lemma
    awk -F '\t' '{print $3}' $file > $high_dir/$lang-dev.tags
done

# Preprocess the test data (identical for both settings), save reference for evaluation
for lang in $(cat $main_dir/language-list.txt)
do
    file=$main_dir/orig_data/answers/task1/$lang-uncovered-test
    awk -F '\t' '{print $3, "<sep>"}' $file | sed 's/;/ /g' > $file.tags
    awk -F '\t' '{print $1}' $file | sed 's/ /⌀/g' | sed 's/./& /g' | sed 's/ $//g'  > $file.lemma
    paste -d ' ' $file.tags $file.lemma > $high_dir/$lang-test.src
    awk -F '\t' '{print $2}' $file | sed 's/ /⌀/g' | sed 's/./& /g' | sed 's/ $//g' > $high_dir/$lang-test.trg
    cp $file $high_dir/$lang-test.ref
    awk -F '\t' '{print $1}' $file > $high_dir/$lang-test.lemma
    awk -F '\t' '{print $3}' $file > $high_dir/$lang-test.tags
done

cp $high_dir/*{src,trg,tags,lemma,ref} $medium_dir

cp $main_dir/orig_data/evaluation/evalm.py $script_dir
rm -rf $main_dir/orig_data/
