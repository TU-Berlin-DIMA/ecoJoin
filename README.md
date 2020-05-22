# Conducting an energy measure

Start processing on jetson:<br>
```sudo python efficient-gpu-joins/impl/benchmark/zmq_bench.py [raspberry ip] [settings file] 0```

Start measure on raspberry pi:<br>
```python efficient-gpu-joins/impl/benchmark/zmq_measure.py```

## Results
Measurement results are stored on the jetson in:<br>
```python efficient-gpu-joins/impl/measures/[current timestamp]```

Results given by stream join executable: ```[setting]bench.csv```
Results given by power measurement executable: ```exp[setting]```

## Settings
Setting files store configurations used to start an experiment. Each line represents one experiment.<br>
```efficient-gpu-joins/impl/benchmark/settings/```

```./bin/gpu_stream:
Usage:
  -n NUM   number of tuples to generate for stream R
  -N NUM   number of tuples to generate for stream S
  -O FILE  file name to write result stream to
  -r RATE  tuple rate for stream R (in tuples/sec)
  -R RATE  tuple rate for stream S (in tuples/sec)
  -w SIZE  window size for stream R (in seconds)
  -W SIZE  window size for stream S (in seconds)
  -p [cpu, gpu]  processing mode (cpu or gpu)
  -s SEC  idle window time
  -S SEC  process window time
  -T enable sleep time window
  -t sleep control in worker
  -b NUM  batchsize for stream R
  -B NUM  batchsize for stream S
  -g NUM  GPU gridsize
  -G NUM  GPU blocksize
  -f enable frequency by stream join
  -e end when worker ends
  ```

## Postprocessing
Postprocessing includes generating energy plots and calculating the average energy consumption.<br>
Running postprocessing scripts:<br>
```efficient-gpu-joins/impl/postprocess/postprocess.sh efficient-gpu-joins/impl/measures/[timestamp]```
