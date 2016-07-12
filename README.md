Demo scripts from [my talk](https://docs.google.com/presentation/d/1PnJoNG0mwTAAo2niOVm-oLfYSM9szYfal-3Ehi0W_GY/edit?usp=sharing)
on graph-based machine learning.

## Setup

Each demo can be run with Python 2.7 or 3.4+. Other versions may also work,
but I haven't tested them.

To install the required packages, run:

    pip install graphs pandas

Installing `graphs` will pull in all the other required dependencies for you.

Note: `pandas` is only required for demo #2, so if you're not going to run
the college clustering then you can skip installing it.


## Demo 1: Swiss Roll Graph

    python demo1-swissroll.py

Three figures will show up, plus an HTML file that can be opened locally with
a web browser.


## Demo 2: College Clustering

    python demo2-clustering.py

Be sure to edit the `SCORECARD_FILE` variable at the top of the file to point
at the location of the college scorecard data on your machine.

This one will take a decent amount of time, depending on how beefy your
computer is. It'll print out some messages along the way to let you know it's
not dead, though, and will eventually show three figures.


## Demo 3: Graph Construction Methods

    python demo3-construction.py

Two figures will appear, each with a 3x3 grid of subplots.


## Bonus: k-Nearest Neighbor Animation

    python generate-knn-gif.py

This one depends on the command-line utility `convert`, which is part of the
ImageMagick library. If you don't want to bother with that, it'll still output
each frame as a PNG file and you can stitch them together as you see fit.
