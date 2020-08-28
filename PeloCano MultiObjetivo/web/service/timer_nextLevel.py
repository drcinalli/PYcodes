__author__ = 'pablonsilva'

from threading import Thread
import time

class LevelThread(Thread):

    def __init__ (self):
        Thread.__init__(self)
        self.service = None

    def run(self):
        #self.service.start_game()
        print 'run'

class GameThread(Thread):

    def __init__ (self):
        Thread.__init__(self)
        self.service = None

    def run(self):
        time.sleep(300000)
        self.service.canReadResult = True

class StartGameThread(Thread):

    def __init__ (self):
        Thread.__init__(self)
        self.service = None

    def run(self):
        print 'O game ira comecar em 30s!'
        time.sleep(10)
        self.service.start_game()