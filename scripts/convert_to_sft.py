"""Convert incremental DST GRPO-format data to ms-swift SFT format.

GRPO format (input):
  {"messages": [...system, user], "audios": [...], "solution": "...", ...}

SFT format (output):
  {"messages": [...system, user, assistant], "audios": [...]}

The assistant turn contains the solution text.

Usage:
  python scripts/convert_to_sft.py \
      --input data/dapo_train.jsonl \
      --output data/sft_train.jsonl

  python scripts/convert_to_sft.py \
      --input data/dapo_val.jsonl \
      --output data/sft_val.jsonl
"""

import argparse
import json
import sys


def convert_line(line: str) -> str:
    """Convert a single GRPO-format JSON line to SFT format."""
    data = json.loads(line)

    messages = data['messages']
    solution = data['solution']
    audios = data.get('audios', [])

    # Append assistant response
    sft_messages = messages + [{'role': 'assistant', 'content': solution}]

    sft_record = {'messages': sft_messages}
    if audios:
        sft_record['audios'] = audios

    return json.dumps(sft_record, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description='Convert GRPO data to SFT format')
    parser.add_argument('--input', required=True, help='Input GRPO JSONL file')
    parser.add_argument('--output', required=True, help='Output SFT JSONL file')
    args = parser.parse_args()

    count = 0
    with open(args.input, 'r', encoding='utf-8') as fin, \
         open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sft_line = convert_line(line)
            fout.write(sft_line + '\n')
            count += 1

    print(f'Converted {count} samples: {args.input} -> {args.output}')


if __name__ == '__main__':
    main()
