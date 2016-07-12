from __future__ import print_function, division
import numpy as np
import os
import pandas as pd
import re
from collections import Counter
from graphs.construction import neighbor_graph
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

# Edit this to the location of the scorecard CSV from the data jam.
SCORECARD_FILE = fname = os.path.expanduser(
    '~/Data/college_stats/scorecard/Most-Recent-Cohorts-Scorecard-Elements.csv')

# Uninformative words to exclude from the university names word clouds
STOP_WORDS = set([
    'and', 'of', 'the', 'The', 'in', 'at', 'for', 'de', '',
    'College', 'Institute', 'University', 'School'
])


def main():
  x, labels = prepare_data(SCORECARD_FILE)
  print('%d schools w/ %d features each' % x.shape)

  # plot k-means clusters over the first two PCs
  pcs = PCA(n_components=2).fit_transform(x)
  y_kmeans = KMeans(n_clusters=4).fit_predict(x)
  clouds_kmeans = word_clouds(y_kmeans, labels)

  _, ax = plt.subplots(figsize=(14, 6))
  scatter_labeled(pcs, labels, ax=ax, colors=y_kmeans, cmap='Dark2',
                  color_labels=clouds_kmeans, edgecolor='none')
  ax.set_title('Top 2 PCs, k-means labels')
  ax.xaxis.set_ticks([])
  ax.yaxis.set_ticks([])

  # build a kNN graph
  print('Building a 10-NN graph, based on cosine distance...')
  dist = pairwise_distances(x, metric='cosine')
  knn = neighbor_graph(dist, precomputed=True, k=10).symmetrize(method='max')
  knn = knn.from_adj_matrix(knn.matrix(csr=True))  # XXX: hack to sparsify
  print(knn.connected_components(return_labels=False), 'connected components')

  # compute some statistics
  apsp = knn.shortest_path(directed=False, unweighted=True)
  eccen = apsp.max(axis=0)
  d = eccen.argmax()
  print('diameter = %d: "%s" <-> "%s"' % (eccen.max(), labels[d],
                                          labels[apsp[d].argmax()]))
  print('radius = %d: "%s"' % (eccen.min(), labels[eccen.argmin()]))

  # find a spectral clustering
  print('Computing spectral clustering...')
  y_spectral = knn.cluster_spectral(9)
  y_spectral = np.argsort(np.argsort(-np.bincount(y_spectral)))[y_spectral]
  clouds_spectral = word_clouds(y_spectral, labels)

  # plot the new clustering over a 2d Isomap embedding
  print('Embedding to 2d with Isomap...')
  emb = knn.isomap(num_dims=2, directed=False)
  _, ax = plt.subplots(figsize=(14, 6))
  knn.plot(emb, ax=ax, directed=False, weighted=False, edge_style='k-',
           vertex_style=dict(marker=',', c='k', s=1, zorder=0))
  scatter_labeled(emb, labels, ax=ax, colors=y_spectral, cmap='Set1',
                  color_labels=clouds_spectral, zorder=2, edgecolor='none')
  ax.set_title('Isomap embedding, spectral clustering labels')
  ax.xaxis.set_ticks([])
  ax.yaxis.set_ticks([])

  # plot the reordered distance matrix
  order = np.argsort(y_spectral)
  _, ax = plt.subplots(figsize=(8, 8))
  imshow_labeled(dist[order][:,order], labels[order], y_spectral, ax=ax)
  ax.set_title('Pairwise cosine distance matrix')

  plt.show()


def word_clouds(y, labels, length=10):
  num_clusters = y.max() + 1  # y is integers from 0 to n-1
  return [','.join(w for w,_ in top_words(labels[y==i]).most_common(length))
          for i in range(num_clusters)]


def top_words(labels, stop_words=STOP_WORDS):
  top = Counter()
  for name in labels:
    for word in re.split(r'[ &?/-]', name):
      if word not in stop_words:
        top[word] += 1
  return top


def scatter_labeled(data, labels, colors=None, color_labels=None, ax=None,
                    **kwargs):
  def format_coord(x, y):
    idx = np.linalg.norm(data - (x,y), axis=1).argmin()
    return labels[idx]

  if ax is None:
    ax = plt.gca()
  ax.format_coord = format_coord

  artist = ax.scatter(*data.T, c=colors, **kwargs)

  if color_labels is None:
    return
  colors = artist.cmap(np.linspace(0, 1, len(color_labels)))
  proxies = [Patch(color=c, label=l) for c,l in zip(colors, color_labels)]
  ax.legend(handles=proxies, fontsize='small')


def imshow_labeled(data, labels, y, ax=None):
  n = len(labels)

  def format_coord(x, y):
    i, j = map(int, (x, y))
    if 0 <= i < n and 0 <= j < n:
      return '"%s" vs "%s' % (labels[i], labels[j])
    return ''

  if ax is None:
    ax = plt.gca()
  ax.format_coord = format_coord

  ax.imshow(data)
  ticks = [0] + np.cumsum(np.bincount(y)).tolist()
  ticklabels = [str(i) for i in range(len(ticks))]
  ax.xaxis.set_ticks(ticks)
  ax.xaxis.set_ticklabels(ticklabels)
  ax.yaxis.set_ticks(ticks)
  ax.yaxis.set_ticklabels(ticklabels)


def prepare_data(fname):
  df = pd.read_csv(fname, index_col=0, encoding='latin-1')

  # winnow down to all numerics, no NaN values
  numeric_cols = df.columns.to_series().groupby(df.dtypes) \
                   .groups[np.dtype('float64')]
  foo = df[numeric_cols].dropna(axis='index', how='all')
  foo.dropna(axis='columns', thresh=7000, inplace=True)
  foo.dropna(axis='index', how='any', inplace=True)

  # get associated labels
  labels = df.INSTNM[foo.index].values

  # normalize columns to make things comparable
  x = normalize(foo.values, axis=0, norm='max')

  return x, labels

if __name__ == '__main__':
  main()
