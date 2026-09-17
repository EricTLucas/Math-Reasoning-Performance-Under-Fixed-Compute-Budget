import json
import os
from pathlib import Path

INPUT_DIR = Path("raw")     
OUTPUT_FILE = Path("processed/reformatted.jsonl")

def is_valid_answer(answer):
    """
    Valid if answer is a string that can be parsed as an integer.
    """
    try:
        int(answer)
        if int(answer) < 0 or int(answer) >= 100000:
            return False
        return True
    except Exception:
        return False


def load_json_file(path: Path):
    """
    Loads JSON files that contain:
    {
        "test": [ {...}, {...}, ... ]
    }

    or
    {
        "train": [ {...}, {...}, ... ]
    }

    """
    with open(path, "r") as f:
        data = json.load(f)

    # Expecting "test": [...]
    if "test" not in data:
        if "train" not in data:
            raise ValueError(f"JSON file {path} does not contain a 'test' key")
        for row in data["train"]:
            yield row
    else:
        for row in data["test"]:
            yield row


def main():
    new_id = 1

    with open(OUTPUT_FILE, "w") as out:
        for file in sorted(INPUT_DIR.glob("*.json")):
            for row in load_json_file(file):

                if "problem" not in row:
                    problem = row.get("question")
                else:
                    problem = row.get("problem")

                
                answer = row.get("answer")

                # Skip malformed entries
                if problem is None or answer is None:
                    continue
                if not is_valid_answer(answer):
                    continue

                formatted = {
                    "id": new_id,
                    "problem": problem,
                    "answer": answer
                }

                out.write(json.dumps(formatted) + "\n")
                new_id += 1

    print(f"Wrote {new_id - 1} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
