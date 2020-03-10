#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 09:42:24 2020

@author: danielcinalli
"""

import array


#number of people in the room
num_pessoas=20

#array of integers
pessoas = array.array('i',(i+1 for i in range(0,num_pessoas)))
print (pessoas.tolist())


#array
pessoas = []
for i in range(num_pessoas):
       pessoas.append(i+1)
       
print (pessoas)
