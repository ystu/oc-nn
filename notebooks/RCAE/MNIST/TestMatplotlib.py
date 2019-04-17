#import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

x = np.random.randn(60)
y = np.random.randn(60)

plt.scatter(x, y, s=20)

plt.show()
