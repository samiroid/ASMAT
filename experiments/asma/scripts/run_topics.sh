set -e
topic_datasets="duggan-1 duggan-2 duggan-3 cameron-2 cameron-1 flood-1"

for ds in $topic_datasets; 
do
    ./experiments/asma/scripts/supervised_ASMA.sh $ds topics.txt
    echo $ds >> "done.txt"
done