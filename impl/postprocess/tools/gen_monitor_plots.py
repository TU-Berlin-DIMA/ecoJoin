import pandas as pd
import plotly
import sys, glob, time
import plotly.express as px
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("usage: python ./gen_energy_plot [path]")
    exit(0)

path = sys.argv[1]
for filename in glob.glob(path +'/exp*'):
    print(filename)
    df = pd.read_csv(filename, sep=' ',comment='#')
    
    fig = px.line(df,y = 'W', title=filename[:-3], range_y=[0,8])
    #fig.show()
    fig.write_image(filename[:-3] + ".png")
    
    #ax = df.plot.bar( y="W")
    #fig = ax.get_figure()
    #fig.savefig(filename[:-3] +".png")
    #input("Press Enter to continue...")

