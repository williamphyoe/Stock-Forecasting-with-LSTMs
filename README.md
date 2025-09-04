## Intro
Project Students
- Chandler Burgess, clb015210
- Wai Yan Phyoe, wxp220006

## Setting up Python Environment
This program requires Python >= 3.9 to run. I used Anaconda Python on Windows myself.

First, create and activate a python virtual environment.
```
python -m venv lstmillions
.\lstmillions\scripts\activate
```
For Anaconda on Mac:
```
conda create --name lstmillions
conda activate lstmillions
```

Next, install all the program dependencies.

`pip install -r requirements.txt`


## Running the program
The program will:
1. Download the data it needs using yfinance
2. Preprocess the data and calculate technical indicator features
3. Run the model with all combinations of paramters
4. Output graphs for all models and the best model for each target

Our paper used AAPL as the ticker, but if you want to use a smaller dataset that runs much quicker we recommend GSIW.

```
#python lstmillions_eval.py <ticker>
#python lstmillions_eval.py aapl
python lstmillions_eval.py gsiw
```


## Cleaning up
To clean up, deactivate the virutal environment and delete its folder.

```
deactivate
rd /s /q lstmillions  # If on Windows
# rm -rf lstmillions  # If using *nix
```

For Anaconda on Mac:
```
conda deactivate
conda remove --name lstmillions --all
```
