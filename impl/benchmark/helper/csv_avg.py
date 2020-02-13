import glob
import csv

with open('result.csv', 'w', newline='') as f_output:
    csv_output = csv.writer(f_output)

    for filename in glob.glob('run*.csv'):
        print (filename)

        with open(filename, newline='') as f_input:
            csv_input = csv.reader(f_input, delimiter=' ')
            header = next(csv_input)
            averages = []

            for col in zip(*csv_input):
                try:
                    averages.append(sum(float(x) for x in col) / len(col))
                except:
                    pass

        csv_output.writerow([filename] + averages)
