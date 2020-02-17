# Conducting a energy measure

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

## Postprocessing
Postprocessing includes generating energy plots and calculating the average energy consumption.<br>
Running postprocessing scripts:<br>
```efficient-gpu-joins/impl/postprocess/postprocess.sh efficient-gpu-joins/impl/measures/[timestamp]```
