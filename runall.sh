#!/bin/bash
echo "Starting test"
count=0
FILE="config.py"
if [ -f "./_papers/test_all.txt" ]; then
    rm "./_papers/test_all.txt"
fi

TEST=("1" "2" "3" "4" "5")

for val1 in "${TEST[@]}"; do
        LINE_NUM=9
        NEW_CONTENT="BACKUP\t\t\t\t = \"${count}_test_all\""
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"

        LINE_NUM=10
        NEW_CONTENT="SAVE\t\t\t\t = \"${count}_test_all_T\""
        sed -i "${LINE_NUM}s/.*/$NEW_CONTENT/" "$FILE"

        python main.py --mode train
        python main.py --mode eval
        last_line=$(tail -n 1 ./RESULTS/${count}_test_all/${count}_test_all.csv | tr -d '\r')
        echo "$last_line" | cat -A
        last_line="$(date +"%T"), $last_line"
        echo "$last_line" >> ./_papers/test_all.txt
        
        count=$((count + 1))
done