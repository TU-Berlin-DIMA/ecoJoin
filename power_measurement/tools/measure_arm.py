from subprocess import PIPE, run
import pandas as pd
import sys
import time

def out(command):
    result = run(command, stdout=PIPE, stderr=PIPE, universal_newlines=True, shell=True)
    return result.stdout

secs = 5
cmd = "timeout " + str(secs) + " ./arm-probe/arm-probe/arm-probe -C ./arm-probe/arm-probe/config"


if len(sys.argv) != 2:
    print("argv != 2")
    print(str(sys.argv))
    exit(0)


text_file = "../measurements/"+sys.argv[1]
with open(text_file, 'r') as f:
    i=0
    o=[]
    for x in f.readlines():
        i = i+1
        if i==1:
            o = [x]
            continue 

        input("Press Enter to continue...")
        l=0
        
        time.sleep(2)
        while l == 0:
            try:
                csv = out(cmd)
                text_file_ = open("/tmp/tmp.csv", "w")
                text_file_.write(csv)
                text_file_.close()
                df = pd.read_csv("/tmp/tmp.csv",comment='#', sep='\s+')
                l=1
            except: 
                pass
            
        o = o + [x.strip() + str(df["V5.0(W)"].mean()) + '\n']
        #o.join([x.strip(), str(df["V5.0(W)"].mean()), '\n'])
        #print(df["V5.0(W)"].mean())
with open(text_file, 'w') as f:
    f.writelines(o) 
