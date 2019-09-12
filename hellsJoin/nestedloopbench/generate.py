import random
import sys
import csv

num = int(sys.argv[2])
count = 0
lines = []
for i in range(num):
    #count = count + random.randint(0,1)
    #lines.append([random.randint(1,21), count, random.randint(1,21)])
    lines.append([random.randint(1,25001), count, random.randint(1,21)])

with open(sys.argv[1], "wb") as f:
    writer = csv.writer(f)
    writer.writerows(lines)
