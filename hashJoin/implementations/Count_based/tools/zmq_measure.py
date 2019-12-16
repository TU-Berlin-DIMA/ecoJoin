import zmq
import subprocess

if len(sys.argv) != 2:
    print("usage: python ./zmq_measure [ip]")

context = zmq.Context()

#  Socket to talk to server
print("Connecting…")
socket = context.socket(zmq.REQ)
socket.connect("tcp://" + ip +":5555")

# Start 
p = subprocess.Popen(['sleep', '5'], stdout=subprocess.PIPE, 
                       shell=True, preexec_fn=os.setsid) )

print("Sending request..")
socket.send(b"Hello")

#  Get the reply.
print("Waiting for stop signal")
message = socket.recv()

os.killpg(os.getpgid(p.pid), signal.SIGTERM)
