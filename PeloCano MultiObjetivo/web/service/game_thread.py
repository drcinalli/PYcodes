__author__ = 'pablonsilva'

from threading import Thread
import time
from stoppableThread import StoppableThread

class GameThread(StoppableThread):

    def __init__ (self):
        Thread.__init__(self)
        self.service = None
        self.Experiment = None

    def run(self):
        print "RUNRUNRUN!"
        self.Experiment = self.service.Experiment
        while self.service.currentLevel <= int(self.service.Experiment.numLevels):
            while self.service.GA.GetStatus() != 'R':
                time.sleep(1)
                print "Aguardando o GA executar!"

            if self.Experiment.id != self.service.Experiment.id:
                return ""

            chromossomes = self.service.GA.GetFront()#self.service.currentLevel) ##PASSAR NIVEL
            population = self.service.GA.GetPopulation()
            fitness = self.service.GA.GetFitnessFront()

            #if self.service.currentLevel > 1:
            #   self.service.processPartialRank(chromossomes,fitness)

            self.service.prepareComparisons(chromossomes,population,fitness)

            t = 0
            while not self.service.canReadResult and t <= 120:
                t += 1
                time.sleep(1)
                print str(t) + "Aguardando Todas as Respostas!"

            if self.Experiment.id != self.service.Experiment.id:
                return ""

            self.service.processResult()

            self.service.processPartialRank(chromossomes,fitness)

            self.service.currentLevel += 1

        print "GA Finalizado!"
        self.service.currentLevel = self.Experiment.numLevels + 1
        chromossomes = self.service.GA.GetFront()
        population = self.service.GA.GetPopulation()
        fitness = self.service.GA.GetFitnessFront()

        self.service.prepareComparisons(chromossomes,population,fitness)

        t = 0
        while not self.service.canReadResult and t <= 120:
            t+= 1
            time.sleep(1)
            print str(t) + "Aguardando Todas as Respostas Finais!"

        self.service.processPartialRank(chromossomes,fitness)


        self.service.currentLevel = 0

        #F -> rank
        while self.service.GA.GetStatus() != 'F':
                time.sleep(1)
                print "Aguardando Rank!"

        self.service.experimentStarted = False
