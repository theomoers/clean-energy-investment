# Clean Energy Investment Causal Forest Analysis

> April 2024 <br>
> Columbia University, SIPA <br>

<br>
Data is taken from the Clean Investment Monitor (Rhodium Group): https://www.cleaninvestmentmonitor.org/database/

## Setup
```bash
git clone https://github.com/theomoers/clean-energy-investment.git
cd clean-energy-investment
# activate env
pip install -r requirements.txt
```

## Running the analysis
The code is developed for jupyter notebooks. It can also be run using the .py files but there are some longer processes that take quite long. It is easier to circumvent those by using the notebooks. <br>
<br>
- data_exploration.ipynb/.py includes the code for the introduction of the paper. <br>
- political_analysis.ipynb/.py includes the code for a preliminary political analysis of partisan dominance in each congressional district from 2018-2024. <br>
- data_cleaning.ipynb/.py includes the code for the assembly of the data set (downloading NASA POWER, ACS district data, and Clean Investment Monitor data). <br>
- causal_forest_implementation.ipynb/.py includes the code for the causal forest, the trimming of the data set, the weighting, calibration, optimization, and validation of the causal forest, and the plots and statistics for the result section. It also includes the DiD regression. <br>

### Key dependencies
```numpy, pandas, matplotlib, seaborn, geopandas, cartopy, us, scikit-learn, econml, statsmodels, linearmodels, scipy, tqdm, requests, censusdata, textwrap```


