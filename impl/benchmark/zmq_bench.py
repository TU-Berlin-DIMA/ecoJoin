import zmq
import sys
import subprocess, os, signal
import datetime

def start_bench(line):
    print("Send start measure signal")
    socket.send(b"Start")
    message = socket.recv()

    # Start GPU Monitor
    # p = subprocess.Popen(['./helper/gpu_monitor.sh'], stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid) 

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

print("Start benchmark")
f = open(setting,'r')

linenumber = 0
for line in f:
    print("linenumber " + str(linenumber))
    if linenumber < start_at:
        linenumber = linenumber + 1
    else:
        linenumber = linenumber + 1
        
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

socket.send(b"End")
