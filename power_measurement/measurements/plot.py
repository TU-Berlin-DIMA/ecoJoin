import pandas as pd
import sys
import matplotlib.pyplot as plt

df = pd.read_csv(sys.argv[1],index_col=0)
pl = df.plot()
pl.set_ylim(bottom=0)
plt.show()
