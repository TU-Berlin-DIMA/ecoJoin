import pandas as pd
import plotly
import sys, glob, time
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go

if len(sys.argv) != 2:
    print("usage: python ./gen_energy_plot [path]")
    exit(0)

fig = go.Figure()
path = sys.argv[1]
for filename in glob.glob(path+ '*.csv'):
    if ".png" not in filename:
        print(filename)

        df = pd.read_csv(filename, sep=' ',comment='#')
        
        fig.add_trace(go.Scatter(y=df['V5.0(W)'], mode='lines', name='lines'))
        #fig = px.line(df, y = 'V5.0(W)', title=filename[:-3], range_y=[0,8])
        #fig = px.line(df,y = 'V5.0(W)', title=filename, range_y=[0,8])
        #fig.show()
        #fig.write_image(filename[:-3] + ".png")
        
        #ax = df.plot.bar( y="W")
        #fig = ax.get_figure()
        #fig.savefig(filename[:-3] +".png")
        #input("Press Enter to continue...")
fig.update_yaxes(range=[0,8])
fig.update_layout(title=path)
fig.write_image(path+ "multiline_plt.png")
