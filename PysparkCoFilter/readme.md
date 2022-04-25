To run this code, simply: 
1. download all files and folders in this direcotry (`./`) to ilab jupyter hub
2. download `transactions_train.csv` from google drive and save to `./co_filter` folder
3. If you want to do a test run, simply open `recommendations.py`, go to line `45` or line `4` of function `cleanData()`, uncomment the `.head(1000)`
4. You can change `1000` to any amount of row you want to use as testing
5. open a terminal at current directory `./`, and run `python recommendations.py`
6. If error caused by memory overflow, simply rerun the code, it will continue where it left off
