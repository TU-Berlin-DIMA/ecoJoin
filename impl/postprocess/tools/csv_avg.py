import glob
import csv, sys

path = sys.argv[1]

with open('result.csv', 'w') as f_output:
    csv_output = csv.writer(f_output)

    for filename in glob.glob(path + 'exp*'):
        print (filename)

        if ".png" not in filename:
            with open(filename) as f_input:
                csv_input = csv.reader(f_input, delimiter=' ')
                
                # ignore comments
                for i in range(10):
                    next(csv_input)
                
                header = next(csv_input)
                averages = []

                for col in zip(*csv_input):
                    try:
                        averages.append(sum(float(x) for x in col) / len(col))
                    except:
                        pass
                print (averages)

            csv_output.writerow([filename] + averages)
