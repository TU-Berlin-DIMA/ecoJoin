import zmq
import subprocess, signal
import sys
import os, time

prog = "/home/pi/efficient-gpu-joins/power_measurement/tools/arm-probe/arm-probe/arm-probe -d 5"

if len(sys.argv) != 1:
    print("usage: python ./zmq_measure")
    exit()

context = zmq.Context()

#  Socket to talk to server
print("Connecting..")
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


while(True):
    #  Get the reply.
    print("Waiting for start signal")
    try:
        message = socket.recv()
    except: # possible measure problem
        socket.send(b"Restart")
        print("Restart")
        continue

    if (message == "Start"):
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
            socket.send(fl)
        print("measure sended")
    else:
        break
