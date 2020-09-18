__author__ = 'pablonsilva'

from game_thread import GameThread
#from polls.interface import InterfaceGA
import random
from datetime import datetime
from models import Experiment, Play, Player

class Game(object):
    INSTANCE = None
    def __init__(self):

        self.gameUsers = [] #users registered to play
        self.minNumPlayers = 1 #minimum number of players to start a game
        self.gameStarted = False #tells if there is enough players to start a game
        self.numLevels = 5 #number of game levels
        self.experimentStarted = False #tells if the experiment is started or not
        self.experimentHasPlayers = False
        self.currentLevel = 1
        self.canReadResult = False
        self.GA = ""
        self.levelReadToPlay = 0
        self.userPlays = None
        self.Experiment = None
        self.numAnswersCurrentLevel = 0
        self.lenFront = 0

        #Teste do rogerio
        self.levelFAKE = -1

    #singleton
    @classmethod
    def get_instance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = Game()
        return cls.INSTANCE


    def start_experiment(self,experiment,ga):
    #start an experiment
        if self.experimentStarted:
            return False,-1

        #self.numLevels = numLevels
        #self.numMinPlayers = numMinPlayes

        #Registrar Experiment no BD
        #exp = Experiment(name = "Experiment_" + str(datetime.now()), numLevels=self.numLevels, numMinPlayers = self.minNumPlayers, start = datetime.now(), time_elapsed_end = -1)
        #exp.save()


        #self.GA.fakeStartEvolution()
        self.Experiment = experiment
        self.experimentStarted = True
        self.GA = ga
        self.numLevels = self.Experiment.numLevels
        return True,self.Experiment.id

    def register_user(self,user):
    #Registers the user that is waiting to play
        #verify if the user is already at the users list.
        if not self.gameUsers.__contains__(user) and self.experimentStarted:
            self.gameUsers.append(user)


        # TEM QUE ARRUMAR ISSO!
        if len(self.gameUsers) >= self.minNumPlayers and not self.gameStarted:
            self.start_game()

    #Starts the game
    def start_game(self):
        self.gameStarted = True

        t = GameThread()
        t.service = Game.get_instance()
        t.start()

        #self.answers = [[ 0 for i in range(2) ] for j in range(len(self.gameUsers)) ]

    def setResult(self,username,result):

        user = self.getUserFromGameUsers(username)
        #self.userPlays[i][self.currentLevel].answers = result
        self.numAnswersCurrentLevel+=1

        #exp = Experiment.objects.latest(id)
        result = result.replace("[","").replace("]","").split(",")

        plays = Play.objects.filter(play_player = user, play_experiment = self.Experiment, level = self.currentLevel)

        for i in range(len(plays)):
            plays[i].answer = result[i]
            plays[i].save()

        if self.numAnswersCurrentLevel == len(self.gameUsers):
            #self.processResult()
            self.canReadResult = True


    def getUserFromGameUsers(self,username):
        found = False
        i = 0
        while i < len(self.gameUsers) and not found:
            if self.gameUsers[i].username == username:
                found = True
            else:
                i+=1

        return self.gameUsers[i]


    def processResult(self):
        answers = [[ 0 for i in range(2) ] for j in range(self.lenFront) ]
        #for i in range(len(self.gameUsers)):
            #user = self.gameUsers[i]
            #exp = Experiment.objects.latest(id)
        plays = Play.objects.filter(play_experiment = self.Experiment, level = self.currentLevel)
        for j in range(len(plays)):
            if plays[j].answer != -1:
                if plays[j].answer == 0:
                    correct = plays[j].chromosomeTwoIndex #verifica o indice da resposta correta
                    incorrect = plays[j].chromosomeOneIndex
                else:
                    correct = plays[j].chromosomeOneIndex #verifica o indice da resposta correta
                    incorrect = plays[j].chromosomeTwoIndex

                answers[correct][0] += 1
                answers[incorrect][1] += 1

        self.GA.SetComparisonsResult(answers)
        self.GA.ContinueEvolution()
        self.canReadResult = False
        self.numAnswersCurrentLevel = 0


    def prepareComparisons(self,chromossomes,population):

        self.createPlays(chromossomes,population)
        self.lenFront = len(chromossomes)

        self.levelReadToPlay += 1


    def getArea(self):
        return self.GA.GetArea()


    def getPlays(self,user):
        user_plays = []

        if self.levelReadToPlay == self.currentLevel:
            #user,i = self.getUserFromGameUsers(username)
            #userPlay = self.userPlays[i][self.currentLevel-1]

            #exp = Experiment.objects.latest(id)
            user_plays = Play.objects.filter(play_player = user).filter(level = self.currentLevel).filter(play_experiment = self.Experiment)

        return user_plays


    def readyToPlay(self):
        return self.levelReadToPlay


    def createPlays(self,chromossomes,population):
        for i in range(len(self.gameUsers)):
            # Distribui os chromossomos entre os jogadores
            for j in range(5):
                a = random.randint(0,len(chromossomes)-1)
                b = -1
                isDifferent = True
                while isDifferent:
                    b = random.randint(0,len(chromossomes)-1)
                    if a != b:
                        isDifferent = False
                play = Play(answer = -1, level = self.currentLevel, chromosomeOne = chromossomes[a], chromosomeOneIndex = a,
                            chromosomeTwo = chromossomes[b], chromosomeTwoIndex = b,play_player = self.gameUsers[i], play_experiment = self.Experiment, answer_time = -1, points = -1)
                play.save()


    def processRank(self,front):

        points = [100,80,60,40,20]

        for i in range(self.Experiment.numLevels):
            point = points[i-1];
            plays = Play.objects.filter(play_experiment = self.Experiment, level = i)
            for p in plays:
                if p.answer != -1:
                    if p.answer == 0:
                        chromossome = p.chromosomeOne
                    else:
                        chromossome = p.chromosomeTwo

                    if chromossome in front:
                        p.points = point
                    else:
                        p.points = 0


    def getRank(self):

        #essa query esta ruim mas funciona
        plays = Play.objects.filter(play_experiment = self.Experiment)

        players = []

        for p in plays:
            if not players.__contains(p.play_player):
                players.append(p.play_player)

        rank = [[ 0 for i in range(2) ] for j in range(len(players))]

        i=0
        for player in players:
            points = 0
            plays = Play.objects.filter(play_experiment = self.Experiment, play_player = player)
            for p in plays:
                points += p.points

            rank[i][0] = p.username
            rank[i][1] = points
            i+=1

        return rank


    def fake(self):
        isReady = self.levelFAKE
        self.levelFAKE+= 1

        if self.levelFAKE == 6:
            self.levelFAKE = 0
        return self.levelFAKE