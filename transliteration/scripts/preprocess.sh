#!/bin/sh
# Distributed under MIT license

# This script preprocesses NEWS2015 shared task datasets.

script_dir=`dirname $0`
main_dir=$script_dir/..
data_dir=$main_dir/data

# Create a list of all the script pairs
for folder in $data_dir/*; do echo ${folder##*/} ; done > $main_dir/language-list.txt

# Split the original train data into train / dev and use the original dev data as test
# Create character-level format for input and XML files for evaluation
for lang in $(cat $main_dir/language-list.txt)
do
    python scripts/extract_by_ids.py --folder $data_dir/$lang
done
