#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 17:10:21 2020

@author: danielcinalli
"""

#
# import math
# number = float(input(" Please Enter any numeric Value : "))
#
# #call function
# squareRoot = math.sqrt(number)
#
# print("The Square Root of a Given Number {0}  = {1}".format(number, squareRoot))
#
# x = int(input('Enter a number : '))
# # The guess answer
# guess = 0.0
# # Used for accuracy. See the condition in While Loop
# epsilon = 0.01
# #used to increment our guess 'ans'
# step = epsilon**2
#
# total_guesses = 0
#
# # We will understand this condition during code analysis
# while (abs(guess**2 - x)) >= epsilon:
#     guess += step
#     total_guesses += 1
#
# print('Total guesses were ' + str(total_guesses))
# if abs(guess**2-x) >= epsilon:
#     print('Failed on square root of ' + str(x))
# else:
#     print(str(guess) + ' is close to the square root of ' + str(x))
#

x = int(input('Enter a number : '))

epsilon = 0.01
left = 0
right = x
# guess start on X/2
guess = (right+left)/2.0

total_guesses = 0

while abs(guess**2 - x) > epsilon:
    print (guess)
    if guess**2 < x:
        left = guess
    else:
        right = guess
    guess = (right+left)/2.0
    total_guesses += 1

print('Total guesses were ' + str(total_guesses))
print(str(guess) + ' is close to the square root of ' + str(x))
