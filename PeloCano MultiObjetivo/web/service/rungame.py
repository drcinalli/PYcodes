__author__ = 'pablonsilva'

from game_thread import GameThread
#from polls.interface import InterfaceGA
import random
from datetime import datetime
from models import Experiment, Play, Player
from timer_nextLevel import StartGameThread

class Game(object):
    INSTANCE = None
    def __init__(self):

        self.gameUsers = [] #users registered to play
        self.gameStarted = False #tells if there is enough players to start a game
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
        self.CanGetRank = False
        self.numMinPlayers = 1
        self.threadGame = None
        self.levelThread = None

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
        #if self.experimentStarted:
        #    return False,-1

        #self.numLevels = numLevels
        #self.numMinPlayers = numMinPlayes

        #Registrar Experiment no BD
        #exp = Experiment(name = "Experiment_" + str(datetime.now()), numLevels=self.numLevels, numMinPlayers = self.minNumPlayers, start = datetime.now(), time_elapsed_end = -1)
        #exp.save()


        self.gameUsers = [] #users registered to play
        self.gameStarted = False #tells if there is enough players to start a game
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
        self.CanGetRank = False

        self.threadGame = None
        self.levelThread = None

        #self.GA.fakeStartEvolution()
        self.Experiment = experiment
        self.experimentStarted = True
        self.GA = ga
        self.numMinPlayers = int(self.Experiment.numMinPlayers)
        return True,self.Experiment.id

    def register_user(self,user):
    #Registers the user that is waiting to play
        #verify if the user is already at the users list.
        if not self.gameUsers.__contains__(user) and self.experimentStarted:
            self.gameUsers.append(user)


        # TEM QUE ARRUMAR ISSO!
        if len(self.gameUsers) >= self.numMinPlayers and not self.gameStarted:
            sg = StartGameThread()
            sg.service = Game.get_instance()
            sg.start()
            #self.start_game()

    #Starts the game
    def start_game(self):
        self.gameStarted = True

        #t = GameThread()
        if self.threadGame is not None:
            self.threadGame.stop()
        self.threadGame = GameThread()
        self.threadGame.service = Game.get_instance()
        self.threadGame.start()

        #self.answers = [[ 0 for i in range(2) ] for j in range(len(self.gameUsers)) ]

    def setResult(self,username,result):

        #if self.levelThread is not None:
        #    self.levelThread.stop()
        #self.levelThread = LevelThread()
        #self.levelThread.service = Game.get_instance()
        #self.levelThread.start()

        try:
            user = self.getUserFromGameUsers(username)
            #self.userPlays[i][self.currentLevel].answers = result

            #exp = Experiment.objects.latest(id)
            result = result.replace("[","").replace("]","").split(",")

            plays = Play.objects.filter(play_player = user, play_experiment = self.Experiment, level = self.currentLevel)

            if plays[0].answer == -1:

                for i in range(len(plays)):
                    plays[i].answer = result[i]
                    plays[i].save()

                self.numAnswersCurrentLevel+=1

            else: return "Resultado ja cadastrado!"

            if self.numAnswersCurrentLevel == len(self.gameUsers):
                #self.processResult()
                self.canReadResult = True
        except:
            return False
        finally:
            return True

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


    def prepareComparisons(self,chromossomes,population,fitness):

        self.createPlays(chromossomes,population,fitness)
        self.lenFront = len(chromossomes)

        self.levelReadToPlay += 1
        print "Atualizando level pronto para jogar:" +str(self.levelReadToPlay)


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


    def createPlays(self,chromossomes,population,fitness):
        users = self.gameUsers

        if self.currentLevel == self.Experiment.numLevels +1:
            users = self.getTopPlayersInternal(3)

        for i in range(len(users)):
            # Distribui os chromossomos entre os jogadores
            for j in range(5):
                a = random.randint(0,len(chromossomes)-1)
                b = -1
                isDifferent = True
                while isDifferent:
                    b = random.randint(0,len(chromossomes)-1)
                    if a != b:
                        isDifferent = False
                play = Play(answer = -1, level = self.currentLevel, chromosomeOne = chromossomes[a],
                            chromosomeOneIndex = a, chromosomeTwo = chromossomes[b], chromosomeTwoIndex = b,
                            play_player = self.gameUsers[i], play_experiment = self.Experiment, answer_time = -1,
                            points = -1,fit_custoOne = fitness[a][0], fit_prodOne = fitness[a][1],
                            fit_custoTwo = fitness[b][0], fit_prodTwo = fitness[b][1])
                play.save()


    def processPartialRank(self,front,fitness):

        medProd = 0
        medCusto = 0

        minProd = fitness[0][1]
        minCusto = fitness[0][0]
        maxProd = fitness[0][1]
        maxCusto = fitness[0][0]

        for i in range(len(fitness)):
            medProd += fitness[i][0]
            medCusto += fitness[i][1]
            if fitness[i][0] < minCusto:
                minCusto = fitness[i][0]
            elif fitness[i][0] > maxCusto:
                maxCusto = fitness[i][1]

            if fitness[i][1] < minProd:
                minProd = fitness[i][1]
            elif fitness[i][1] > maxProd:
                maxProd = fitness[i][1]

        medProd /= len(fitness)
        medCusto /= len(fitness)

        medProdNorm = (medProd - minProd) / (maxProd - minProd)
        medCustoNorm = (medCusto - minCusto) / (maxCusto - minCusto)

        plays = Play.objects.filter(play_experiment = self.Experiment, level = (self.currentLevel), play_player__type = "H")
        for p in plays:
            prod = 0
            custo = 0
            if p.answer != -1:
                if p.answer == 0:
                    prod = p.fit_prodOne
                    custo = p.fit_custoOne
                else:
                    prod = p.fit_prodTwo
                    custo = p.fit_custoTwo

            prodNorm = (prod - minProd) / (maxProd - minProd)
            custoNorm = (custo - minCusto) / (maxCusto - minCusto)

            points = 0
            points += self.getPoints(prodNorm,medProdNorm)
            points += self.getPoints(custoNorm,medCustoNorm)

            #points += (1 / abs(prodNorm - medProdNorm)) * 100
            #points += (1 / abs(custoNorm - medCustoNorm)) * 100

            p.points = points

            p.save()

        self.CanGetRank = True

    def getPoints(self,a,med):
        point = (1 / abs(a - med))

        if point <= 0.05:
            point *= 300
        elif point <= 0.10:
            point *= 250
        elif point <= 0.20:
            point *= 200
        elif point <= 0.40:
            point *= 100
        else:
            point *= 25

        return point


    def getRankTotalGeral(self,username):

        #essa query esta ruim mas funciona
        plays = Play.objects.filter(play_user__username = username)

        players = []

        for p in plays:
            if not players.__contains__(p.play_player):
                players.append(p.play_player)

        rank = [[ 0 for i in range(2) ] for j in range(len(players))]

        i=0
        for player in players:
            points = 0
            plays = Play.objects.filter(play_player = player)
            for p in plays:
                points += p.points

            rank[i][0] = player.username
            rank[i][1] = points
            i+=1

        return rank

    def getPartialRank(self,username):

        latest_exp = Experiment.objects.latest('id')

        #essa query esta ruim mas funciona
        plays = Play.objects.filter(play_experiment = latest_exp, play_player__type = 'H')

        players = self.gameUsers

        for p in plays:
            if not players.__contains__(p.play_player):
                players.append(p.play_player)

        rank = [[ 0 for i in range(2) ] for j in range(len(players))]

        i=0
        for player in players:
            points = 0
            plays = Play.objects.filter(play_experiment = latest_exp,play_player = player)
            for p in plays:
                if p.points != -1:
                    points += p.points

            rank[i][0] = player.username
            rank[i][1] = points
            i+=1

        print rank
        rankSorted = sorted(rank, key=self.getKey, reverse=True)
        print rankSorted

        ranking = []
        for i in range(len(rankSorted)):
            ranking.append([i+1,rankSorted[i][0],rankSorted[i][1]])

        rankFinal = []


        #TEm que melhorar isso. Mas funciona.
        contains = False
        for i in range(len(ranking)):
            if ranking[i][1] == username:
                contains = True
            if i < 5:
                rankFinal.append(ranking[i])

            if contains:
                break
            else:
                if ranking[i][1] == username:
                    rankFinal.append(ranking[i])

        return rankFinal

    def getKey(self,item):
        return item[1]

    def getRank(self,username):

        latest_exp = Experiment.objects.latest('id')

        #essa query esta ruim mas funciona
        plays = Play.objects.filter(play_experiment = latest_exp)

        players = []

        for p in plays:
            if not players.__contains__(p.play_player):
                players.append(p.play_player)

        rank = [[ 0 for i in range(2) ] for j in range(len(players))]

        i=0
        for player in players:
            points = 0
            plays = Play.objects.filter(play_experiment = latest_exp, play_player = player)
            for p in plays:
                points += p.points

            rank[i][0] = p.play_player.username
            rank[i][1] = points
            i+=1

        return rank

    def canGetRank(self):
        return self.CanGetRank

    def fake(self):
        isReady = self.levelFAKE
        self.levelFAKE+= 1

        if self.levelFAKE == 6:
            self.levelFAKE = 0
        return self.levelFAKE

    def getOnline(self):

        players = []
        for user in self.gameUsers:
            players.append(user.username)

        return players

    def getTopPlayers(self,num):

        latest_exp = Experiment.objects.latest('id')

        #essa query esta ruim mas funciona
        plays = Play.objects.filter(play_experiment = latest_exp, play_player__type = 'H')

        players = self.gameUsers

        for p in plays:
            if not players.__contains__(p.play_player):
                players.append(p.play_player)

        rank = [[ 0 for i in range(2) ] for j in range(len(players))]

        i=0
        for player in players:
            points = 0
            plays = Play.objects.filter(play_experiment = latest_exp,play_player = player)
            for p in plays:
                if p.points != -1:
                    points += p.points

            rank[i][0] = p.play_player.username
            rank[i][1] = points
            i+=1

        rankSorted = sorted(rank, key=self.getKey, reverse=True)
        print rankSorted

        ranking = []

        n = num
        if num > len(self.gameUsers):
            n = len(self.gameUsers)

        for i in range(n):
            ranking.append([i+1,rankSorted[i][0],rankSorted[i][1]])

        players = []
        for user in self.gameUsers:
            players.append(user.username)

        return players

    def getTopPlayersInternal(self,num):

        latest_exp = Experiment.objects.latest('id')

        #essa query esta ruim mas funciona
        plays = Play.objects.filter(play_experiment = latest_exp, play_player__type = 'H')

        players = self.gameUsers

        for p in plays:
            if not players.__contains__(p.play_player):
                players.append(p.play_player)

        rank = [[ 0 for i in range(2) ] for j in range(len(players))]

        i=0
        for player in players:
            points = 0
            plays = Play.objects.filter(play_experiment = latest_exp,play_player = player)
            for p in plays:
                if p.points != -1:
                    points += p.points

            rank[i][0] = p.play_player.username
            rank[i][1] = points
            i+=1

        rankSorted = sorted(rank, key=self.getKey, reverse=True)
        print rankSorted

        n = num
        if num > len(self.gameUsers):
            n = len(self.gameUsers)

        ranking = []
        for i in range(n):
            ranking.append([i+1,rankSorted[i][0],rankSorted[i][1]])

        players = []
        for user in self.gameUsers:
            players.append(user)

        return players