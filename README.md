## How to clone this repository on your machine?

- Open a terminal and navigate to the folder in which you want to copy this repository using `cd`
- Type `git clone https://github.com/pdonnel/EC_exercise.git`

## How to use the code

You have two python files in the code.  
`EC_source.py` is used to generate data for the absorption of a EC beam in a plasma.
It then stores the data in `.npy` format files.
By default the code runs in a vectorized way. If you want to
force the parallelization to occur you can use the `--parallel` flag.  
`EC_plots.py` uses the data files saved by `EC_source.py` to produces plots.
