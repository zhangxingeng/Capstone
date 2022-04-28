To run this code, simply: 
1. Save `recommendation.py` to a project direcotry (`./`) 
2. Create a foler in `./` called `co_filter` and `cd` into it
3. download all three files to ./co_filter by copy and paste these three commands:
  1. `$ gdown --fuzzy https://drive.google.com/file/d/1huGN_CCrbLEaN0NZJz98rTiYRi0d_vrg/view?usp=sharing`
  2. `$ gdown --fuzzy https://drive.google.com/file/d/1rmYH8JPehUcAcM4Y7Fwg6kiGr4QJc9oo/view?usp=sharing`
  3. `$ gdown --fuzzy https://drive.google.com/file/d/1uu_h9Usoe8tamJk1c4M1qfx6R6pD2yES/view?usp=sharing`
4. Now you should have a folder structure like this:
    ```bash
    ProjectRoot
    ├── co_filter
    │   ├── articles_clean.csv
    │   ├── customers_clean.csv
    │   └── transactions_selected.csv
    └── recommendation.py
    ```
7. If you want to do a test run, simply open `recommendations.py`, go to line `45` or line `4` of function `cleanData()`, add `.head(1000)` to the end of line
8. You can change `1000` to any amount of row you want to use as testing
9. open a terminal at current directory `./`, and run `python recommendations.py`
10. If error caused by memory overflow, simply rerun the code, it will continue where it left off
11. If you got disk quota error, let me know
