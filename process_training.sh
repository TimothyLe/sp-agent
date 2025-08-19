#!/bin/bash

# Directory to scan
DIRECTORY="./datasets"

# Python script to invoke
PYTHON_SCRIPT="src/utils/pdf_convert_llm.py"

# Loop through each file in the directory
for file in "$DIRECTORY"/*; do
    if [ -f "$file" ]; then
        fname="${file%.*}"
        echo "Processing $file"
        python3 "$PYTHON_SCRIPT" "$file" "$fname"
    fi
done

