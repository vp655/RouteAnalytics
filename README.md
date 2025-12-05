# Ravens Route Package

A lightweight, production-ready Python package for evaluating wide receiver routes, generating catch-probability predictions, and animating route-level tracking data. The package ships with a pretrained XGBoost Booster model, a saved route label encoder, a feature specification JSON, and helper functions that make it easy to use the model on any row from your final matchup dataset.

# User Guide (Start Here)

## Installation

Install directly from GitHub via HTTPS:

pip install git+https://github.com/vp655/RavensRoutePackage.git

If in a jupyter notebook the command is:

!pip install git+https://github.com/vp655/RavensRoutePackage.git


This installs the package `ravens-route` (imported as `ravens_route` in Python) and all required dependencies.

## Data Required 

Download `final_matchup_data.csv` from the data folder in the Github repository. 

Then, visit https://www.kaggle.com/competitions/nfl-big-data-bowl-2021/data and sign in to the competition. Download all the data, 
and save it in a folder called data_dir in the same workspace you are currently in. 

## Predict Catch Probability


```python

import pandas as pd
from ravens_route import predict_route_prob

df = pd.read_csv("final_matchup_data.csv")
row = df.iloc[0]  

prob = predict_route_prob(row)
print("Catch Probability:", prob)

```

The function handles:
- Loading the pretrained XGBoost Booster.
- Loading the expected feature list.
- Loading the saved route label mapping.
- Encoding the route value.
- Running the Booster’s predict() method and returning a float in [0, 1].


## Generate a Play Animation

If you are using the animation utilities to visualize a given play:

```python

from ravens_route import animate_play_from_row
import pandas as pd

df = pd.read_csv("final_matchup_data.csv")
row = df.iloc[0]

anim = animate_play_from_row(
    row=row,
    data_dir="data_dir",              # folder containing tracking CSVs from 2021 Big Data Bowl 
    out_gif="animations/example.gif", # output GIF path
    fps=10,
    show=True
)
```

This will:
- Load the corresponding tracking data from data_dir.
- Create a frame-by-frame visualization of the route and coverage.
- Save the GIF to the specified path.
- Optionally display the animation if the environment supports it.

# Package Contents

After installation, the package is available under your Python 3.11 site-packages directory, for example

The structure looks like:

ravens_route/
    __init__.py
    inference.py
    models_io.py
    animation.py  
    models/
        route_model.json
        route_features.json
        route_label_mapping.json
               

- route_model.json: XGBoost Booster model (trained route-level catch probability model).
- route_features.json: Ordered list of feature names used by the model.
- route_label_mapping.json: Mapping from route string (e.g. "GO", "SLANT", "WHEEL") to integer codes.


# IMPORTANT: Dependency Versions

The package has been tested with and currently depends on the following versions (as reported by pip in the environment):

Core dependencies:
- pandas==2.2.0
- numpy==1.26.2
- xgboost==3.0.5

Visualization and image handling (used by animation utilities):
- matplotlib==3.8.2
- pillow==10.1.0

Transitive dependencies (automatically handled by pip):
- scipy==1.12.0 (from xgboost)
- python-dateutil==2.8.2
- pytz==2023.3.post1
- tzdata==2023.4
- contourpy==1.2.0
- cycler==0.12.1
- fonttools==4.47.0
- kiwisolver==1.4.5
- packaging==23.2
- pyparsing==3.1.1
- six==1.16.0

If you install via pip as shown above, these versions (or compatible ones) will be pulled in automatically. For maximum reproducibility, you can pin them explicitly in your own environment.

# Common Errors and Fixes

Below are the most likely issues you or other users may encounter, plus how to fix them.

1) Matplotlib / Animation Issues

Symptoms:
- Errors when generating GIFs or showing plots.
- Inconsistent or broken rendering of animations.
- Backend-related errors when calling animate_play_from_row.

Cause:
Often due to matplotlib version or backend issues. Package not compatible with matplotlib versions >= 3.9.

Fix:
The package has been tested with matplotlib==3.8.2 and pillow==10.1.0. To enforce these versions, run:
pip install "matplotlib==3.8.2" "pillow==10.1.0"

Also ensure:
- You are not accidentally using a very old notebook or IDE backend that conflicts with this version.
- If needed, restart your kernel / Python interpreter after installation or upgrade.

2) macOS OpenMP / libomp Error (XGBoost)

Symptoms:
On macOS, you might see something like:
Library not loaded: @rpath/libomp.dylib

Cause:
XGBoost relies on OpenMP for parallelization. On macOS, you must install libomp manually.

Fix (macOS only):
brew install libomp
brew link libomp --force
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH"

Add the export line to your shell configuration (e.g., ~/.zshrc or ~/.bash_profile), then restart your terminal or IDE.

3) Python Version Incompatibility (Python 3.13+)

Symptoms:
ImportError involving xgboost or pandas internals, or strange behavior when installing xgboost on very new Python versions.

Cause:
Certain xgboost releases (including 3.0.5) are not yet compatible with Python 3.13+.

Fix:
Use Python 3.11 or Python 3.12 for this project. For example:
- On Windows: install Python 3.11 or 3.12 from python.org.
- On macOS/Linux: use pyenv, conda, or your package manager to create an environment with Python 3.11 or 3.12.


# Contact

For internal questions, extensions, or integration into other workflows, please contact:
Vir Pandit (vir.pandit1@gmail.com)
