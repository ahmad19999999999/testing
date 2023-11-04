# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 13:44:53 2023

@author: pc
"""

import pandas as pd
import category_encoders as ce

data = pd.DataFrame({'city': ['delhi','hyderabad','delhi','delhi','gorgon','hyderabad']})
encoder = ce.OrdinalEncoder(cols=['city'])
td= encoder.fit_transform(data)