import csv

"""
    This is a small utility you can use to move the label of a dataset from the first column to the last.
"""
def move_first_column_to_end(input_file, output_file):
    with open(input_file, mode='r', newline='') as infile, \
            open(output_file, mode='w', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            if row:
                reordered = row[1:] + [row[0]]
                writer.writerow(reordered)


# Usage:
if __name__ == '__main__':
    move_first_column_to_end("testFile.csv", "fileFileFixed.csv")