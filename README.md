# ML-4-Trading

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:800/1*_GiyDH1bob96fo8JsO1_hg.png" alt="Sublime's custom image" style="width: 75%; height: auto;"/>
</p>

In this Project, I implemented two strategies and compared their performance. One strategy is a manual strategy, where I developed the trading rules. The other is a strategy learner, which will develop the trading rules using artificial intelligence.

### Description of Files:

StrategyLearner.py 
- Code implementing a StrategyLearner object from RTlearner and BagLearner. This ML Strategy uses bags of Random Forest Learners.
- In the training phase (e.g., add_evidence()) the learner will be provided with a stock symbol and a time period. It uses this data to learn a strategy. For instance, a classification-based learner will use this data to make predictions about future price changes.  
- In the testing phase (e.g., testPolicy()) the learner will be provided a symbol and a date range. All learning should be turned OFF during this phase. 

ManualStrategy.py 
- Constructs a strategy by using the indicator functions in indicators.py. Used SMA, BBP, and Momentum
indicators to determine when to buy/sell.

RTlearner.py
- Contains the code for the regression Random Tree class.

BagLearner.py
- Contains the code for the regression Bag Learner (i.e., a BagLearner containing Random Trees).  

indicators.py
- Houses all indicator functions and can compute portfolio stats.
 
marketsimcode.py
- Main use for this file is calculating the portfolio value given a series of prices. It's leveraged in experiment1.py, experiment2.py, and testproject.py

experiment1.py and experiment2.py  
- Experiment 1 should compare the results of your manual strategy and the strategy learner.
- Experiment 2 conducts an experiment with the StrategyLearner that shows how changing the value of impact should affect in-sample trading behavior.

testproject.py 
- Code initializing/running all necessary files for the report, including experiment1 and experiment2. 


