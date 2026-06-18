#!/bin/bash
cd "$(dirname "$0")"
CONFIG="${1:-server_config.json}"
count=$(python3 -c "import json; c=json.load(open('$CONFIG')); print(len(c.get('instances', [1])))")
for i in $(seq 0 $((count - 1))); do
  python3 translate_server.py --config "$CONFIG" --instance "$i" &
done
wait
