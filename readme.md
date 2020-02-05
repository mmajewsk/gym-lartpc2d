# LARTPC Game

## What is this

This repository is contains an game-like environment made from LARTPC detector readouts.
The goal here is to move a window (eg. 3x3 pixels on image), 
and use the input from the window to categorise each pixel.
Another way to put this task is; you have to redraw a grey image (each pixel has singular float value e.g. [0.7]),
into a colorful image (one of three colors e.g. [0,1,0] or [1,0,0] or [0,0,1]), by steering quadratic brush (e.g. 5x5 pixels)
which has the same position on source image, as well as on canvas.

example `bot.py`:

![](https://i.imgur.com/IyswEwy.gif)

There are three types of windows here:
 - source - an image that we have to base final categories upon, each pixel has a single float value (if y=f(x) then this is x)
 - target - an image that we are trying to recreate, based on source  (and this is y)
 - result - the actual game map (sort of like empty canvas), this where the result of categorisation is saved
 
On the bottom row you can see entire maps/environments/images of the game, 
on the top row, you can see what is visible to the agent in the game (so window of pixels).

## Dependancies

This repository should work on any python 3.7.
The best approach is to create new virtual environment.

## Installation

```bash
git clone https://github.com/mmajewsk/lartpc_game
cd lartpc_game
pip install -r requirements.txt
```

## Pre-running steps
### Getting the data

** This library needs to read data processed from lartpc, which you have to download and process by yourself **
Follow [readme in downloading_data](downloading_data/readme.md) to do all of the steps.


### Setting up path
Assuming that you are in `lartpc_game` folder do:

```
export PYTHONPATH=$(pwd)
```

This will add `lartpc_game` to your python.

## Running `bot.py`

This is a very simple bot-actor, that will move by trying to explore new (non-empty) pixels.
It will also output a random category to the canvas/result map.

e.g.

```
python examples/bot.py ../../assets/dump 
```

Help (you cane use `-h` to see that message)
```
usage: bot.py [-h] [--viz-off] data_path

Runs a simple bot showcasing the game. e.g. usage bot.py ../../assets/dump
bot.py ../../assets/dump --viz-off

positional arguments:
  data_path   Path to the data generated from lartpc

optional arguments:
  -h, --help  show this help message and exit
  --viz-off   Run without visualisation/opencv (helpful for debug)
```

**If you have problem exiting the visualisation**:
first do ctrl-c in your terminal, than press space on the visualisation

## Documentation

The best documentation is to follow the `bot.py` example, 
to figure out how it works, 
or my solution with reinforcement learning.

# Resources