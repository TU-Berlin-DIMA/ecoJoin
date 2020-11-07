import zmq
import sys
import subprocess, os, signal, time
import datetime
import csv

def start_bench(line):
    print("Send start measure signal")
    socket.send(b"Start")
    message = socket.recv()

    # Start GPU Monitor
    #p = subprocess.Popen(['./helper/benchmark_.sh'], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 

    # Start Bench
    subprocess.call([benchmark + " " + line], shell = True, preexec_fn=os.setsid)

    # os.killpg(os.getpgid(p.pid), signal.SIGTERM)

    #  Send reply back to client
    print("End Bench " + line)
    socket.send(b"End Bench")

if len(sys.argv) < 3:
    print("usage: python ./zmq_bench [ip] [setting] [start at line]")
    exit(0)

benchmark = "./helper/benchmark_script.sh"
ip = sys.argv[1] 
setting = sys.argv[2]

start_at = 0
if len(sys.argv) == 4:
    start_at = int(sys.argv[3])

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://" + ip +":5555")

dir_ = "../measures/" 
dir_ += datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
print("create dir " + dir_)
os.mkdir(dir_)

####################################
def ref_measure():
    print("Reference Measure")
    socket.send(b"Reference")
    message = socket.recv()
    time.sleep(15)
    socket.send(b"End Reference")
    message = socket.recv()
    #return float(str(message))
    return float(message)

####################################
print("Start benchmark")
f = open(setting,'r')

config  = []
latency = []
i_time = []
p_time = []
energy = []
output_t = []
input_t = []

refs = []

# Reset Freqs
subprocess.call(["/home/adimon/efficient-gpu-joins/impl/benchmark/helper/reset_freqs.sh"], shell = True, preexec_fn=os.setsid)

linenumber = 0
for line in f:
    print("linenumber " + str(linenumber))
    if linenumber < start_at:
        linenumber = linenumber + 1
    else:
        linenumber = linenumber + 1
        

        refs.append(ref_measure())
        
        setting_stripped = line.replace(" ","").strip()
        print(setting_stripped)
        
        start_bench(line)

        #  Wait for measure file
        print("Wait for measure file")
        message = socket.recv()

        # Check if benchmark measure failed on the measure device
        while (message == "Restart"):
            start_bench(line)
            message = socket.recv()

        f = open(dir_ + '/' +'exp'+ setting_stripped ,'wb')
        f.write(message)
        f.close()

        # Move measure files
        # subprocess.call(["mv sys_monitor.csv " + dir_ + "/" + setting_stripped + "sys_monitor.csv"  ], shell = True)
        subprocess.call(["mv bench.csv " + dir_ + "/" + setting_stripped + "bench.csv"  ], shell = True)


        # Read Latency
        line_ = []
        with open(dir_ + "/" + setting_stripped + "bench.csv") as f:
            for line in f:
                cline = line.split(",")

        config.append(setting_stripped)
        latency.append(float(cline[2]))
        i_time.append(float(cline[5]))
        p_time.append(float(cline[6]))
        output_t.append(float(cline[0]))
        input_t.append(float(cline[3]))


        # calc avg energy
        with open(dir_ + '/' +'exp'+ setting_stripped) as f_input:
            csv_input = csv.reader(f_input, delimiter=' ')

            # ignore comments
            for i in range(9):
                next(csv_input)

            header = next(csv_input)
            s = 0
            i = 0
            for row in csv_input:
                try:
                    s = s + float(row[4])
                except:
                    s = s + 0
                i = i + 1
            s = s / i
            energy.append(s)

        # Reset Freqs
        subprocess.call(["/home/adi/efficient-gpu-joins/impl/benchmark/helper/reset_freqs.sh"], shell = True, preexec_fn=os.setsid)

with open(dir_ + '.csv', 'w') as f_output:
    csv_output = csv.writer(f_output)
    for i in range(len(config)):
        csv_output.writerow([config[i] , latency[i] , i_time[i] , p_time[i] , energy[i] , input_t[i] , output_t[i]] )
        print([config[i] , latency[i] , i_time[i] , p_time[i] , energy[i] , input_t[i] , output_t[i]] )

# Remove Folder
subprocess.call(["rm -r " + dir_], shell = True)
print(refs)
