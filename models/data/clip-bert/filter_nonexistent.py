import json
import os

def main(input_stream, output_stream, file_key: str):
    for line in input_stream:
        parsed = json.loads(line)
        if os.path.isfile(parsed[file_key]):
            output_stream.write(line)


if __name__ == "__main__":
    import argparse
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=sys.stdin, type=argparse.FileType('r', encoding="utf-8"), help="Json lines input")
    parser.add_argument("--output", default=sys.stdout, type=argparse.FileType('w', encoding="utf-8"), help="Json lines output")
    parser.add_argument("--key", default="path")

    args = parser.parse_args()

    main(args.input, args.output, args.key)
