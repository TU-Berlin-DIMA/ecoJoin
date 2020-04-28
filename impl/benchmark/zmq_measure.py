import zmq
import subprocess, signal
import sys
import os, time
import glob
import csv, sys

####################################
def get_average_W(filename):
    with open(filename) as f_input:
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
        print('Average Energy: ' + str(s))
        return (s)

####################################
prog = "/home/pi/efficient-gpu-joins/power_measurement/tools/arm-probe/arm-probe/arm-probe -d 5"

if len(sys.argv) != 1:
    print("usage: python ./zmq_measure")
    exit()

context = zmq.Context()

#  Socket to talk to server
print("Connecting..")
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

####################################
def ref_measure():
    print("Waiting for reference signal")
    message = socket.recv()
    if (not message == b"Reference"):
        print("Expected Message :'Reference'")

    p = subprocess.Popen([prog + " > /tmp/measure.txt"], stdout=subprocess.PIPE,shell=True, preexec_fn=os.setsid)
    socket.send(b"Started")
    message = socket.recv()

    print("Kill measure")
    os.kill(p.pid, signal.SIGTERM)
    for i in range(0,40):
        try:
            os.kill(p.pid+i, signal.SIGTERM)
        except:
            pass
    print(p.pid)

    avg_W = get_average_W("/tmp/measure.txt")
    socket.send(str(float(avg_W)).encode())
    return avg_W


####################################
while(True):
    avg_W = ref_measure()

    #  Get the reply.
    print("Waiting for start signal")
    try:
        message = socket.recv()
    except: # possible measure problem
        socket.send(b"Restart")
        print("Restart")
        continue

    if (message == b"Start"):
        p = subprocess.Popen([prog + " > /tmp/measure.txt"], stdout=subprocess.PIPE,shell=True, preexec_fn=os.setsid)

        socket.send(b"Started")
        time.sleep(3)

        #  Get the reply.
        print("Waiting for stop signal")
        try:
            message = socket.recv()
        except: # possible measure problem
            socket.send(b"Restart")
            print("Restart")
            continue


        print("Kill measure")
        #os.killpg(p.pid, signal.SIGTERM)
        os.kill(p.pid, signal.SIGTERM)
        for i in range(0,40):
            try:
                os.kill(p.pid+i, signal.SIGTERM)
            except:
                pass
        print(p.pid)

        f = open("/tmp/measure.txt",'rb')
        fl = f.read()
        if fl:
            socket.send(b'# avgW '+ str(avg_W).encode('UTF-8') + b'\n' + fl)
        print("measure sended")
    else:
        break
