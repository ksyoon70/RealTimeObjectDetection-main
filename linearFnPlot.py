# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:45:37 2023

@author: headway
"""

import matplotlib.pyplot as plt
import numpy as np

xarr = [0.455, 0.526,0.617,0.291,0.304, 0.769,0.643,0.12]
yarr = [0.43,0.16,0.39,0.44,0.21,0.36,0.12,0.6]

plt.plot(xarr,yarr,'ro')


x = np.linspace(0, 1, 100)
a =-0.38
b = 0.52
plt.plot(x,a*x+b,'b-')
plt.axis([0,1,0,1])
plt.show()