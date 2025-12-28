import json
with open("data/examples.jsonl", 'r') as f:
    with open("data/read.txt", 'w') as out:
        for line in f:
            data = json.loads(line)
            json.dump(data, out, indent=4)
            out.write("\n")