# -*- coding: utf-8 -*-
"""
Created on Thu Jul 10 15:40:33 2025

@author: sb636
"""

import os


KSFolder = 'K+S'


for r, d, f in os.walk(KSFolder):      # r=root, d=directories, f=files

    for file in f:
        if file == 'config.zip' or file == 'logs.zip':

            foo = os.path.join(r, file)
            print(foo)
            os.remove(foo)