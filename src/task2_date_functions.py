
# %% Setup
import sys
# set the path to the repository:
repo_dir = 'C:/DEV/Portfolio/src/'

# append the path to the utility functions
sys.path.append(repo_dir+'/utils/')


# %% 1. Function that takes as input two timestamps of the form 2017/05/13 12:00 and calculates their differences in hours.

# load the function:
from utility import get_difference_in_hours

# test the function:
time_difference = get_difference_in_hours('2017/05/13 12:00', '2018/06/13 16:00')


# %% 2. Expand the above function to only count the time difference between 09:00 – 17:00 and only on weekdays.

# load the function:
from utility import get_difference_in_hours_during_nine_to_five

# test the function:
time_difference = get_difference_in_hours_during_nine_to_five('2017/05/13 12:00', '2018/06/13 16:00')

time_difference = get_difference_in_hours_during_nine_to_five('2023/05/26 16:00', '2023/05/30 8:00')

time_difference = get_difference_in_hours_during_nine_to_five('2023/06/1 16:00', '2023/06/01 8:00')

time_difference = get_difference_in_hours_during_nine_to_five('2023/06/1 16:00', '2023/06/01 16:40')

time_difference = get_difference_in_hours_during_nine_to_five('2023/06/1 16:10', '2023/06/02 16:40')

# %%
