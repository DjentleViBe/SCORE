#!/bin/bash
echo "Starting grid search"

count=0
FILE="config.py"
FFNHIDDEN=("1024" "512")
MAXSEQLEN=("32" "16")
NUMHEADS=("8" ""4)
NUMLAYERS=("6" "3")
DMODEL=("256" "512")
OVERLAP=("8")
LAMBDA=("1.5" "0.5")
ALPHA=("0.1" "0.3")
DELTA=("0.001" "0.005")

if [ -f "gridsearch.txt" ]; then
    rm "gridsearch.txt"
fi

for val1 in "${FFNHIDDEN[@]}"; do
for val2 in "${MAXSEQLEN[@]}"; do
for val3 in "${NUMHEADS[@]}"; do
for val4 in "${NUMLAYERS[@]}"; do
for val5 in "${DMODEL[@]}"; do
for val6 in "${OVERLAP[@]}"; do
for val7 in "${LAMBDA[@]}"; do
for val8 in "${ALPHA[@]}"; do
for val9 in "${DELTA[@]}"; do

        LINE_NUM=9
        NEW_CONTENT="BACKUP\t\t\t\t = \"${count}_grid_search\""
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"

        LINE_NUM=10
        NEW_CONTENT="SAVE\t\t\t\t = \"${count}_grid_search_T\""
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"

        LINE_NUM=28
        NEW_CONTENT="EPOCHS\t\t\t = 10"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"

        LINE_NUM=31
        NEW_CONTENT="FFN_HIDDEN\t\t\t = $val1"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"
        LINE_NUM=32
        NEW_CONTENT="MAX_SEQ_LENGTH\t\t\t = $val2"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"
        LINE_NUM=33
        NEW_CONTENT="NUM_HEADS\t\t\t = $val3"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"
        LINE_NUM=35
        NEW_CONTENT="NUM_LAYERS\t\t\t = $val4"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"
        LINE_NUM=36
        NEW_CONTENT="D_MODEL\t\t\t = $val5"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"
        LINE_NUM=48
        NEW_CONTENT="OVERLAP\t\t\t = $val6"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"
        LINE_NUM=49
        NEW_CONTENT="LAMBDA\t\t\t = $val7"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"
        LINE_NUM=50
        NEW_CONTENT="ALPHA\t\t\t = $val8"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"
        LINE_NUM=51
        NEW_CONTENT="DELTA\t\t\t = $val9"
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"

        python main.py --mode gridsearch
        last_line=$(tail -n 1 ./RESULTS/${count}_grid_search/${count}_grid_search.csv | tr -d '\r')
        echo "$last_line" | cat -A
        last_line="$(date +"%T"), $last_line, $val1, $val2, $val3, $val4, $val5, $val6, $val7, $val8, $val9"
        echo "$last_line" >> gridsearch.txt

        
        count=$((count + 1))
        
done
done
done
done
done
done
done
done
done
