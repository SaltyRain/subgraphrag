import json
from pathlib import Path

def join_json_arrays(file1: Path, file2: Path, output_file: Path) -> None:
    """Join two JSON files (each containing an array) and save the result to a new file."""
    with open(file1, "r", encoding="utf-8") as f1:
        data1 = json.load(f1)
    with open(file2, "r", encoding="utf-8") as f2:
        data2 = json.load(f2)

    # Combine arrays
    combined = data1 + data2

    # Optionally, remove duplicates if needed (based on full object equality)
    # combined = [dict(t) for t in {tuple(sorted(d.items())) for d in combined}]

    with open(output_file, "w", encoding="utf-8") as out:
        json.dump(combined, out, ensure_ascii=False, indent=2)

    print(f"✅ Combined {len(data1)} + {len(data2)} = {len(combined)} records saved to {output_file}")
