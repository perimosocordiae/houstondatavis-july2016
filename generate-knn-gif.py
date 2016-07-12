import matplotlib
import numpy as np
import os
matplotlib.use('Agg')
from glob import glob
from graphs.construction import incremental_neighbor_graph
from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
from subprocess import check_call


def main():
  X, y = make_moons(n_samples=100, shuffle=False, noise=0.1)

  fig, ax = plt.subplots()
  ax.scatter(*X.T, c=y, zorder=2, cmap='seismic', edgecolor='w', lw=1, s=50)
  ax.xaxis.set_ticks([])
  ax.yaxis.set_ticks([])

  for k, g in enumerate(incremental_neighbor_graph(X, k=np.arange(10))):
    g.plot(X, directed=False, weighted=False, ax=ax, edge_style='k-',
           vertex_style=dict(c='k', marker=',', s=1, zorder=0),
           title='k = %d' % k)
    plt.savefig('two_moons-%d.png' % k)

  check_call(['convert', '-delay', '50', 'two_moons-?.png', 'two_moons.gif'])

  for f in glob('two_moons-?.png'):
    os.remove(f)

if __name__ == '__main__':
  main()
