#!/bin/bash

# Vietnamese NL2SQL Local Evaluation Runner

echo "Starting Vietnamese NL2SQL Local Evaluation..."

API_URL="${1:-http://localhost:8000}"
QUERIES_PER_COMPLEXITY="${2:-50}"
OUTPUT_FILE="${3:-./evaluation_results.json}"

echo "API URL: $API_URL"
echo "Queries per complexity: $QUERIES_PER_COMPLEXITY"
echo "Output file: $OUTPUT_FILE"

# Check if API is running
echo "Checking API health..."
curl -s "$API_URL/health" > /dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: API server not responding at $API_URL"
    echo "Please start the API server first:"
    echo "  ./start_api.sh <model_path>"
    exit 1
fi

echo "API server is responding"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run evaluation
echo "Running evaluation..."
python local_evaluation.py \
    --api-url "$API_URL" \
    --queries-per-complexity "$QUERIES_PER_COMPLEXITY" \
    --output "$OUTPUT_FILE"

echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_FILE"
