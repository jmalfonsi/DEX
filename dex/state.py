"""Lossless typed JSON records for a future trained tokenizer/frontend.

No heuristic semantic classifier. Chunking/tokenization are explicitly separate.
"""
from dataclasses import dataclass
import hashlib
import json
import math


@dataclass(frozen=True)
class Record:
    path: str
    kind: str
    text: str

    def cache_key(self, model_hash, tokenizer_hash):
        payload = json.dumps([model_hash, tokenizer_hash, self.path, self.kind, self.text],
                             ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def records(state):
    result = []
    def walk(value, path):
        if isinstance(value, dict):
            if not all(isinstance(k, str) for k in value):
                raise ValueError("JSON object keys must be strings")
            result.append(Record(path, "object", ""))
            for key in sorted(value):
                walk(value[key], path + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            result.append(Record(path, "array", str(len(value))))
            for i, item in enumerate(value):
                walk(item, path + "/" + str(i))
        elif value is None:
            result.append(Record(path, "null", "null"))
        elif isinstance(value, bool):
            result.append(Record(path, "boolean", "true" if value else "false"))
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("nonfinite JSON number")
            result.append(Record(path, "number", json.dumps(value)))
        elif isinstance(value, str):
            result.append(Record(path, "string", value))
        else:
            raise ValueError("unsupported state type")
    walk(state, "")
    return result

