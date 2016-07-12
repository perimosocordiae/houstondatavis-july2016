import numpy as np
from graphs.construction import neighbor_graph
from graphs.generators.swiss_roll import swiss_roll
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def main():
  np.random.seed(1234)
  X, theta = swiss_roll(8, 500, return_theta=True)

  print('Figure 1 of 3: bare coordinates in 3d')
  ax = Axes3D(plt.figure())
  ax.scatter(*X.T, c=theta)

  print('Figure 2 of 3: 5-NN graph in original coordinates')
  g = neighbor_graph(X, k=5).symmetrize('max')
  g.plot(X, directed=False, weighted=False, fig='new', edge_style='k-',
         vertex_style=dict(c=theta))

  print('Writing swiss_roll.html for force-directed layout demo')
  g.to_html('swiss_roll.html', directed=False, weighted=False,
            vertex_colors=theta)

  print('Figure 3 of 3: 2d Isomap embedding of 5-NN graph')
  emb = g.isomap(num_dims=2)
  _, ax = plt.subplots(figsize=(10, 5))
  g.plot(emb, directed=False, weighted=False, ax=ax, edge_style='k-',
         vertex_style=dict(c=theta))
  ax.xaxis.set_ticks([])
  ax.yaxis.set_ticks([])
  plt.show()

if __name__ == '__main__':
  main()
