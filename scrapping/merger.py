import sys
import glob
import os.path as path
import os


def main():
    folder = sys.argv[1]
    output = sys.argv[2]

    if path.exists(output):
        os.remove(output)

    for input_file_path in glob.glob(path.join(folder, "*.tsv")):
        with open(output, "a", encoding="utf-8") as output_f:
            with open(input_file_path, "r", encoding="utf-8") as input_f:
                output_f.writelines([path.splitext(path.basename(input_file_path))[0] + '\t' + line
                                     for line in input_f.readlines()])


if __name__ == "__main__":
    main()
