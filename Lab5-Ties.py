# -*- coding: utf-8 -*-
"""
Lab 5: Tie Production Problem in PuLP

@author: Kevin Jia
"""

import numpy as np
import pandas as pd # We will discuss this more next week!
from pulp import *

# Pandas DataFrame construction
# Take this as given until next Wednesday
TieTypes = ['AllSilk', 'AllPoly', 'PolyCotton', 'SilkCotton']
Profit = pd.Series([16.24, 8.22, 8.77, 8.66], index = TieTypes)
Silk = pd.Series([0.125, 0, 0, 0.066], index = TieTypes)
Polyester = pd.Series([0, 0.08, 0.05, 0], index = TieTypes)
Cotton = pd.Series([0, 0, 0.05, 0.044], index = TieTypes)
MinReq = pd.Series([5000, 10000, 13000, 5000], index = TieTypes)
MaxDem = pd.Series([7000, 14000, 16000, 8500], index = TieTypes)

TieData = pd.DataFrame({'Profit': Profit, 
                              'Silk': Silk, 
                              'Polyester': Polyester, 
                              'Cotton': Cotton, 
                              'MinReq': MinReq,
                              'MaxDem': MaxDem})


# NOT RUN, but to help guide you:
# View all items: TieData
# Get all labels (tie types): TieData.index - can loop over this?
# Get an entire column: TieData['Silk'] 
# Get an entire row: TieData.loc['AllPoly']
# Get information about a particular item: TieData['Silk']['SilkCotton']


# Complete the LP below.





# Solving routines - no need to modify other than slotting your name and username in.
prob.writeLP('Ties.lp')

prob.solve()

print("Name and Username here \n")

# The status of the solution is printed to the screen
print("Status:", LpStatus[prob.status])

# Each of the variables is printed with its resolved optimum value
for v in prob.variables():
    print(v.name, "=", v.varValue)

# The optimised objective function valof Ingredients pue is printed to the screen    
print("Total Profit from Ties = ", value(prob.objective))