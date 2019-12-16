import zmq
import sys
import subprocess, os

if len(sys.argv) != 2:
    print("usage: python ./zmq_bench [benchmark_script]")

benchmark = sys.argv[1] 

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("Wait for Request")
#  Wait for next request from client
message = socket.recv()
print("Received request: %s" % message)

print("Start Benchmark")

# Start Bench
p = subprocess.call([benchmark], shell = True, preexec_fn=os.setsid)

#  Send reply back to client
print("Send end signal")
socket.send(b"End")

#  Wait for measure file
print("Wait for measure file")
message = socket.recv()
f = open("/tmp/measure.csv",'wb')
f.write(message)
f.close()

print("File written")
