# -*- coding: utf-8 -*-
"""
Created on Tue May 22 15:28:08 2018

@author: Jeff
"""

import pandas as pd


d = {'col1':[1,2,3,4,5,6], 'col2':[6,5,4,3,7,2], 'col3':[4,3,5,2,3,4]}
df = pd.DataFrame(d)
df = df[df['col1'] in [1,3,5,7,8]]
print(df)