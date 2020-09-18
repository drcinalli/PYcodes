__author__ = 'pablonsilva'

from threading import Thread
import time

class GameThread(Thread):

    def __init__ (self):
        Thread.__init__(self)
        self.service = None

    def run(self):
        print "RUNRUNRUN!"
        while self.service.currentLevel <= self.service.numLevels:
            while self.service.GA.GetStatus() != 'R':
                time.sleep(1)

            chromossomes = self.service.GA.GetFront()#self.service.currentLevel) ##PASSAR NIVEL
            population = self.service.GA.GetPopulation()

            self.service.prepareComparisons(chromossomes,population)

            while not self.service.canReadResult:
                time.sleep(1)

            self.service.processResult()

            self.service.currentLevel += 1

        #F -> rank
        while self.service.GA.GetStatus() != 'F':
                time.sleep(1)

        self.service.processRank(self.service.GA.GetFront())

        self.service.experimentStarted = False
