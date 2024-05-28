#!/bin/bash

# Check if the user provided a command to run
if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command>"
    exit 1
fi

# Define the output file
OUTPUT_FILE="command_output.log"

# Run the command with time and capture both outputs
{ time "$@"; } 2>&1 | tee "$OUTPUT_FILE"
