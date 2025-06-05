#!/bin/bash

INPUT="$1"
EXEC="$2"

if [ -z "$INPUT" ] || [ -z "$EXEC" ]; then
    echo "Użycie: ./run_dct.sh <plik_wejściowy.jpg> <ścieżka_do_programu>"
    exit 1
fi

# Wyodrębnienie typu z nazwy pliku wykonywalnego
BASENAME=$(basename "$EXEC")
if [[ "$BASENAME" == *"seq"* ]]; then
    TYPE="seq"
elif [[ "$BASENAME" == *"openmp"* ]]; then
    TYPE="omp"
elif [[ "$BASENAME" == *"cuda"* ]]; then
    TYPE="cuda"
elif [[ "$BASENAME" == *"cufft"* ]]; then
    TYPE="cufft"
else
    TYPE="unknown"
fi

# Wyodrębnienie nazwy pliku wejściowego bez rozszerzenia
INPUT_NAME=$(basename "$INPUT")
INPUT_BASE="${INPUT_NAME%.*}"

# mk = 1
echo "mk = 1"
OUTPUT="wynik_${INPUT_BASE}_${TYPE}1.jpg"
"$EXEC" "$INPUT" "$OUTPUT" 1
echo "----------------------------"

# mk = 5, 10, ..., 100
for MK in $(seq 5 5 100)
do
    echo "mk = $MK"
    OUTPUT="wynik_${INPUT_BASE}_${TYPE}${MK}.jpg"
    "$EXEC" "$INPUT" "$OUTPUT" "$MK"
    echo "----------------------------"
done
