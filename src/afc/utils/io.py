import os
import json
from typing import List


def save_to_jsonl(data: List[dict], path: str, encoding="utf-8", ensure_ascii=False):
    with open(path, "w", encoding=encoding) as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=ensure_ascii)
            f.write("\n")


def load_jsonl(filepath: str, encoding="utf-8", ensure_ascii=False):
    output = []

    with open(filepath, "r", encoding=encoding) as f:
        for line in f:
            record = json.loads(line)
            output.append(record)

    return output
