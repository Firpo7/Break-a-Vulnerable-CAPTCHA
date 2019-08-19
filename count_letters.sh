#!/bin/bash

echo "TRAIN SET"
tot=0
for c in $(grep -o . <<< "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
do
    a=`ls "train_set/$c/" | wc -l`;
    echo "$c -> $a";
    tot=$(($tot+$a));
done;

echo -e "\nTrain Set Total -> $tot\n\n";


echo "TEST SET"
tot=0
for c in $(grep -o . <<< "ABCDEFGHIJKLMNOPQRSTUVWXYZ");
do
    a=`ls "test_set/$c/" | wc -l`;
    echo "$c -> $a";
    tot=$(($tot+$a));
done;

echo -e "\nTest Set Total -> $tot";
