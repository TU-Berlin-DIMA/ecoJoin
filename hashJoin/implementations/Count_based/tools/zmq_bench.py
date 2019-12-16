import zmq
import sys
import subprocess

if len(sys.argv) != 2:
    print("usage: python ./zmq_bench [benchmark_script]")

benchmark = sys.argv[1] 

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

while True:
    #  Wait for next request from client
    message = socket.recv()
    print("Received request: %s" % message)
    
    print("Start Benchmark")

    # Start Bench
    subprocess.call([benchmark], shell = True)

    #  Send reply back to client
    socket.send(b"End")
