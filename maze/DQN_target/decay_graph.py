import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


eps = 1.0
eps_end = 0.01
eps_decay = 0.99992
e = []
for i in range(100_000):
    eps = max(eps_end, eps_decay*eps)
    # if eps == eps_end:
    #print(i, eps)
    e.append(eps)

plt.plot(np.arange(len(e)), e)
plt.show()
