# fairness-web

Platform for evaluating the fairness and identifying potential descrimination issues in datasets or machine learning models. 
This demo is implemented by Ziqi Wang from the Research Institute of Trustworthy Autonomous Systems, Southern University of Science and Technology.
## Before running
### Install required packages
```
pip install -r requirement.txt
```

Verified versions:
* numpy-1.21.2
* pandas-1.3.2
* torch-1.9.0
* flask-2.0.1
* pyecharts-1.9.0

## How to run
1. Run the application 
```
python app.py
```

2. Copy the ip shown in the terminal and paste it into a web browser (Chrome is recommended, we have not tested other browsers).

## Samples
* Sample dataset can be found in "test_cases/data".
* Sample models can be found in "test_cases/model".
