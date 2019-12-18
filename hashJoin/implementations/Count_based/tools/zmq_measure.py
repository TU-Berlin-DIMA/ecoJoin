import zmq
import subprocess, signal
import sys
import os

prog = "~/efficient-gpu-joins/power_measurement/tools/arm-probe/arm-probe/arm-probe"

if len(sys.argv) != 2:
    print("usage: python ./zmq_measure [ip]")
    exit()

ip = sys.argv[1]
context = zmq.Context()

#  Socket to talk to server
print("Connecting..")
socket = context.socket(zmq.REQ)
socket.connect("tcp://" + ip +":5555")

# Start 
p = subprocess.Popen([prog + " >> /tmp/measure.txt"], stdout=subprocess.PIPE,shell=True, preexec_fn=os.setsid)
print(p.pid)
print("Sending request..")
socket.send(b"Hello")

#  Get the reply.
print("Waiting for stop signal")
message = socket.recv()

print("Kill measure")
os.killpg(p.pid, signal.SIGTERM)

f = open("/tmp/measure.txt",'rb')
fl = f.read()
if fl:
    socket.send(fl)
print("measure sended")
