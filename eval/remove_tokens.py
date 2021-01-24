import os
import re
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str)
    args = parser.parse_args()
    if os.path.exists(args.text):
        fi = open(args.text, "r", encoding="utf-8")
        fo = open(args.text+"_2", "w", encoding="utf-8")
        for line in fi:
            cleaned = re.sub(r"( <EOS>| <unk>)", "", line.strip())
            fo.write(cleaned+"\n")
        fi.close()
        fo.close()
