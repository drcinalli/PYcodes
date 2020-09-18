from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change it
    in any way you see fit.
  """
  
  ANDAR=0
  MOV=''
    
  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.
    
    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()
    
    #Aqui chama a funcao de Avaliacao
    #o retorno e uma posicao legal
    if self.ANDAR == 0:
        movimento = self.evaluationFunction(gameState, legalMoves)
    else:
        if self.MOV in legalMoves:
            movimento =self.MOV
            self.ANDAR = self.ANDAR -1
        else:
            self.ANDAR = 0 
            movimento = self.evaluationFunction(gameState, legalMoves)
            
        
     
    # Choose one of the best actions
#    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
#    bestScore = max(scores)
#    #caso tenha score negativo
#    bestNegScore = -999
#    for i in range(len(scores)) :
#        if scores[i] <0 and scores[i] > bestNegScore:
#            bestNegScore =scores[i]
    
    #bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    #chosenIndex = random.choice(bestIndices)

#    if bestNegScore > -999:
#        bestIndices = [index for index in range(len(scores)) if scores[index] == bestNegScore]
#    else: 
#        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    
#    chosenIndex = random.choice(bestIndices)
    
            
    "Add more of your code here if you want to"
    #print(chosenIndex)
    print "---------------------"
    print(legalMoves)
    print movimento
    return movimento
  
  def evaluationFunction(self, currentGameState, legalMoves):
    """
    Design a better evaluation function here. 
    
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.
    """
  # Useful information you can extract from a GameState (pacman.py)
    returnScore=float(0)
    #successorGameState = currentGameState.generatePacmanSuccessor(action)
    #newPos = successorGameState.getPacmanState().getPosition()
    pos = currentGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates() 
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    # aqui comeca o bagulho
    
    #declaracao de variaveis  booleanas
    comida_perto=0   # <= 5
    comida_longe=0   # >5
    
    fantasma_perto=0  # <= 3
    fantasma_longe=0  # >3
    
    medo_baixo=0 # <=3
    medo_alto=0 # >3
    
    #bolota_perto = 0 # <=4
    #bolota_longe = 0 # >4
    
    #declaracao de variaveis  discretas
    dist_comida=0
    dist_fantasma=0
    medo=0
    
    #flag_comida_range_curto=0 # quantidade <= range 6
    #flag_comida_range_longo=0 # quantidade > range 6 && <=10
    
    #descobrir a distancia da comida mais proxima
    ListComida=oldFood.asList()
    ListComida.sort(lambda x,y: util.manhattanDistance(pos, x)-util.manhattanDistance(pos, y))
    DistanciaComida=util.manhattanDistance(pos, ListComida[0])
    
    #setar valor booleano para comida
    if(DistanciaComida<=5 ):
        comida_perto= 1
    if(DistanciaComida>=4 and DistanciaComida<=34 ):
        comida_longe = 1
        
    dist_comida = DistanciaComida
    #fim setor dist_comida
    
        
    #setar valor das distancias do fantasma
    PosicaoFantasma=[Ghost.getPosition() for Ghost in newGhostStates]
    if len(PosicaoFantasma) ==0 : 
        fantasma_perto=1
        dist_fantasma = 0       
    else: 
        PosicaoFantasma.sort(lambda x,y: disCmp(x,y,pos))
        DistanciaFantasma = util.manhattanDistance(pos, PosicaoFantasma[0])
        
        if(DistanciaFantasma<=4):
            fantasma_perto=1   
        elif(DistanciaFantasma>=3 and DistanciaFantasma<=30):
            fantasma_longe=1
        
        dist_fantasma = DistanciaFantasma
    #fim setor dis_fantasma
          
    #setar valor da invencibilidade do PacMan (medo_fantasma)
    medo = min(newScaredTimes)
    
     #setar valor booleano para medo
    if(medo<=4):
        medo_baixo= 1
    elif(medo>=3):
        medo_alto = 1
        
    #fim setor medo


    #Variaveis de desfuzzyficacao
    RiscoAlto=[]
    RiscoBaixo=[]
    Agressivo=[]
    
    #inferencia
    if comida_perto and fantasma_perto and medo_baixo:
        resultado_fuzzy=Fugir_Fantasma(DistanciaFantasma,PosicaoFantasma[0],pos,currentGameState, legalMoves)
        #Calcula a paertinencia VALOR, MAXIMO do Grafico, MINIMO do Grafico, INVERTE ou nao o valor
        risco = min(Pertinencia(DistanciaComida,5,0,1),Pertinencia(DistanciaFantasma,4,0,1))
        agressividade = Pertinencia(medo,4,0,1)
        
        RiscoAlto.append(risco)
   
    if comida_perto and fantasma_perto and medo_alto:
        resultado_fuzzy=Atacar_Fantasma(DistanciaFantasma,PosicaoFantasma[0],pos,currentGameState, legalMoves)
        #Calcula a paertinencia VALOR, MAXIMO do Grafico, MINIMO do Grafico, INVERTE ou nao o valor
        risco = min(Pertinencia(DistanciaComida,5,0,1),Pertinencia(DistanciaFantasma,4,0,1))
        agressividade = Pertinencia(medo,9,3,0)

        Agressivo.append(agressividade)

    if comida_perto and fantasma_longe and medo_baixo:
        resultado_fuzzy=Comer_Comida(DistanciaComida,ListComida[0],pos, currentGameState, legalMoves)
        #Calcula a paertinencia VALOR, MAXIMO do Grafico, MINIMO do Grafico, INVERTE ou nao o valor
        risco = min(Pertinencia(DistanciaComida,5,0,1),Pertinencia(DistanciaFantasma,9,3,0))
        agressividade = Pertinencia(medo,4,0,1)

        RiscoBaixo.append(risco)

    if comida_perto and fantasma_longe and medo_alto:
        resultado_fuzzy=Comer_Comida(DistanciaComida,ListComida[0],pos, currentGameState, legalMoves)
        #Calcula a paertinencia VALOR, MAXIMO do Grafico, MINIMO do Grafico, INVERTE ou nao o valor
        risco = min(Pertinencia(DistanciaComida,5,0,1),Pertinencia(DistanciaFantasma,9,3,0))
        agressividade = Pertinencia(medo,9,3,0)

        RiscoBaixo.append(risco)

    if comida_longe and fantasma_perto and medo_baixo:
        resultado_fuzzy=Fugir_Fantasma(DistanciaFantasma,PosicaoFantasma[0],pos,currentGameState, legalMoves)
        #Calcula a paertinencia VALOR, MAXIMO do Grafico, MINIMO do Grafico, INVERTE ou nao o valor
        risco = min(Pertinencia(DistanciaComida,9,4,0),Pertinencia(DistanciaFantasma,4,0,1))
        agressividade = Pertinencia(medo,4,0,1)
 
        RiscoAlto.append(risco)
   
    if comida_longe and fantasma_perto and medo_alto:
        resultado_fuzzy=Atacar_Fantasma(DistanciaFantasma,PosicaoFantasma[0],pos,currentGameState, legalMoves)
        #Calcula a paertinencia VALOR, MAXIMO do Grafico, MINIMO do Grafico, INVERTE ou nao o valor
        risco = min(Pertinencia(DistanciaComida,9,4,0),Pertinencia(DistanciaFantasma,4,0,1))
        agressividade = Pertinencia(medo,9,3,0)

        Agressivo.append(agressividade)
    
    if comida_longe and fantasma_longe and medo_baixo:
        resultado_fuzzy=Comer_Comida(DistanciaComida,ListComida[0],pos, currentGameState, legalMoves)
        #Calcula a paertinencia VALOR, MAXIMO do Grafico, MINIMO do Grafico, INVERTE ou nao o valor
        risco = min(Pertinencia(DistanciaComida,9,4,0),Pertinencia(DistanciaFantasma,9,3,0))
        agressividade = Pertinencia(medo,4,0,1)
 
        RiscoBaixo.append(risco)
   
    if comida_longe and fantasma_longe and medo_alto:
        resultado_fuzzy=Comer_Comida(DistanciaComida,ListComida[0],pos, currentGameState, legalMoves)
        #Calcula a paertinencia VALOR, MAXIMO do Grafico, MINIMO do Grafico, INVERTE ou nao o valor
        risco = min(Pertinencia(DistanciaComida,9,4,0),Pertinencia(DistanciaFantasma,9,3,0))
        agressividade = Pertinencia(medo,9,3,0)

        RiscoBaixo.append(risco)

    ##DEFUZZYFICACAO
    # pega a maior pertinencia
    maxRiscoAlto = 0 
    maxRiscoBaixo = 0 
    maxAgressividade = 0 
    
    if len(RiscoAlto) >0: 
        RiscoAlto.sort(cmp=None, key=None, reverse=True)
        maxRiscoAlto=RiscoAlto[0]
    if len(RiscoBaixo) >0: 
        RiscoBaixo.sort(cmp=None, key=None, reverse=True)
        maxRiscoBaixo=RiscoBaixo[0]

    if len(Agressivo) >0: 
        Agressivo.sort(cmp=None, key=None, reverse=True)
        maxAgressividade= Agressivo[0]
    

    if (maxRiscoAlto >= maxRiscoBaixo and maxRiscoAlto >= maxAgressividade):
        resultado_fuzzy_2=Fugir_Fantasma(DistanciaFantasma,PosicaoFantasma[0],pos,currentGameState, legalMoves)
    
    elif (maxRiscoBaixo >= maxRiscoAlto and maxRiscoBaixo >= maxAgressividade):
        resultado_fuzzy_2=Comer_Comida(DistanciaComida,ListComida[0],pos, currentGameState, legalMoves)
        #aqui faz o pacman andar direto QDO NAO TEM RISCO
        self.ANDAR = 0
        self.MOV = resultado_fuzzy_2

        
    elif (maxAgressividade >= maxRiscoAlto and maxAgressividade >= maxRiscoAlto):
        resultado_fuzzy_2=Atacar_Fantasma(DistanciaFantasma,PosicaoFantasma[0],pos,currentGameState, legalMoves)
    
    
    #aqui faz o pacman andar direto QDO NAO TEM RISCO
    
    
    
    #return resultado_fuzzy

    return resultado_fuzzy_2


def Pertinencia(valor,  maximo, minimo, inv): 
    valor = valor - minimo
    maximo = maximo - minimo
    
    resultado = float(valor)/float(maximo)
    if inv==1: 
        resultado = 1- resultado
    
    return resultado

def Comer_Comida(distancia, comida, pos,currentGameState, legalMoves): 


    retorno = 'STOP'

    #calcula os movimentos ate que a distancia entre a comida mais perto seja a menor.
    for action in legalMoves:
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanState().getPosition()
        
        #com o movimento, calcula a distancia NOVA
        pseudo_nova_dist = util.manhattanDistance(newPos, comida)
        #caso a distancia tenha diminuido, otimo, esse e o mocimento.
        if pseudo_nova_dist <= distancia:
            retorno = action
            distancia = pseudo_nova_dist
         
    
    #nunca deve entrar aqui
    return retorno       
        

def Atacar_Fantasma(distancia,fantasma,pos,currentGameState, legalMoves): 


    retorno = 'STOP'

    #calcula os movimentos ate que a distancia entre a pacman e o fantasma seja a menor.       
    for action in legalMoves:
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanState().getPosition()
        
        #com o movimento, calcula a distancia NOVA
        pseudo_nova_dist = util.manhattanDistance(newPos, fantasma)
        #caso a distancia tenha diminuido, otimo, esse e o mocimento.
        if pseudo_nova_dist <= distancia:
            retorno = action
            distancia = pseudo_nova_dist
         
       
    return retorno

def Fugir_Fantasma(distancia,fantasma,pos,currentGameState, legalMoves): 


    retorno = 'STOP'

    #calcula os movimentos ate que a distancia entre a pacman e o fantasma seja a menor.       
    for action in legalMoves:
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanState().getPosition()
        
        #com o movimento, calcula a distancia NOVA
        pseudo_nova_dist = util.manhattanDistance(newPos, fantasma)
        #caso a distancia tenha diminuido, otimo, esse e o mocimento.
        if pseudo_nova_dist >= distancia:
            retorno = action
            distancia = pseudo_nova_dist
         
       
    return retorno
    
    
def disCmp(x,y,newPos):
    if (util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))<0: return -1
    else: 
        if (util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))>0: return 1
        else:
            return 0
def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This abstract class** provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    
    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
    
    **An abstract class is one that is not meant to be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.  
  """
  
  def __init__(self, evalFn = scoreEvaluationFunction):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = evalFn
  def setDepth(self, depth):
    """
      This is a hook for feeding in command line argument -d or --depth
    """
    self.depth = depth # The number of search plies to explore before evaluating
    
  def useBetterEvaluation(self):
    """
      This is a hook for feeding in command line argument -b or --betterEvaluation
    """
    print("I was here")
    betterEvaluationFunction.firstCalled=True;
    self.evaluationFunction = betterEvaluationFunction
    

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth 
      and self.evaluationFunction.
    """
    "*** YOUR CODE HERE ***"
    numOfAgent=gameState.getNumAgents();
    trueDepth=numOfAgent*self.depth
    LegalActions=gameState.getLegalActions(0)
    if Directions.STOP in LegalActions: 
        LegalActions.remove(Directions.STOP)
    listNextStates=[gameState.generateSuccessor(0,action) for action in LegalActions ]
    #print(self.MiniMax_Value(numOfAgent,0,gameState,trueDepth))
    v=[self.MiniMax_Value(numOfAgent,1,nextGameState,trueDepth-1) for nextGameState in listNextStates] 
    MaxV=max(v)
    listMax=[]
    for i in range(0,len(v)):
        if v[i]==MaxV:
             listMax.append(i)
    i = random.randint(0,len(listMax)-1)
    
    print(LegalActions)
    print(v)
    print(listMax)
    action=LegalActions[listMax[i]]
    return action

  def MiniMax_Value(self,numOfAgent,agentIndex, gameState, depth):
      LegalActions=gameState.getLegalActions(agentIndex)
      listNextStates=[gameState.generateSuccessor(agentIndex,action) for action in LegalActions ]
      if (gameState.isLose() or gameState.isWin() or depth==0): 
              return self.evaluationFunction(gameState)
      else:
          if (agentIndex==0):
              return max([self.MiniMax_Value(numOfAgent,(agentIndex+1)%numOfAgent,nextState,depth-1) for nextState in listNextStates] )
          else :
              return min([self.MiniMax_Value(numOfAgent,(agentIndex+1)%numOfAgent,nextState,depth-1) for nextState in listNextStates])

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def Alpha_Beta_Value(self, numOfAgent, agentIndex, gameState, depth, alpha, beta):
      LegalActions=gameState.getLegalActions(agentIndex)
      if (agentIndex==0): 
         if Directions.STOP in LegalActions: 
             LegalActions.remove(Directions.STOP)
      listNextStates=[gameState.generateSuccessor(agentIndex,action) for action in LegalActions ]
      
      # terminal test      
      if (gameState.isLose() or gameState.isWin() or depth==0): 
              return self.evaluationFunction(gameState)
      else:
          # if Pacman
          if (agentIndex == 0):
              v=-1e308
              for nextState in listNextStates:
                  v = max(self.Alpha_Beta_Value(numOfAgent, (agentIndex+1)%numOfAgent, nextState, depth-1, alpha, beta), v)
                  if (v >= beta):
                      return v
                  alpha = max(alpha, v)
              return v
          # if Ghost
          else:
              v=1e308
              for nextState in listNextStates:
                  v = min(self.Alpha_Beta_Value(numOfAgent, (agentIndex+1)%numOfAgent, nextState, depth-1, alpha, beta), v)
                  if (v <= alpha):
                      return v
                  beta = min(beta, v)
              return v
              
  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    numOfAgent=gameState.getNumAgents();
    trueDepth=numOfAgent*self.depth
    LegalActions=gameState.getLegalActions(0)
    
    # remove stop action from list of legal actions
    if Directions.STOP in LegalActions: 
        LegalActions.remove(Directions.STOP)
    
    listNextStates = [gameState.generateSuccessor(0,action) for action in LegalActions ]
    
    # check whether minimax value for -l minimaxClassic are 9, 8 , 7, -492
    # print(self.Alpha_Beta_Value(numOfAgent,0,gameState,trueDepth))
    
    # as long as beta is above the upper bound of the eval function
    v = [self.Alpha_Beta_Value(numOfAgent,1,nextGameState,trueDepth-1, -1e308, 1e308) for nextGameState in listNextStates] 
    MaxV=max(v)
    listMax=[]
    for i in range(0,len(v)):
        if v[i]==MaxV:
             listMax.append(i)
    i = random.randint(0,len(listMax)-1)
    
    print(LegalActions)
    print(v)
    print(listMax)
    action=LegalActions[listMax[i]]
    return action

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  def Expectimax_Value(self,numOfAgent,agentIndex, gameState, depth):
      LegalActions=gameState.getLegalActions(agentIndex)
      listNextStates=[gameState.generateSuccessor(agentIndex,action) for action in LegalActions ]
      if (gameState.isLose() or gameState.isWin() or depth==0): 
              return self.evaluationFunction(gameState)
      else:
          if (agentIndex==0):
              return max([self.Expectimax_Value(numOfAgent,(agentIndex+1)%numOfAgent,nextState,depth-1) for nextState in listNextStates] )
          else :
              listStuff=[self.Expectimax_Value(numOfAgent,(agentIndex+1)%numOfAgent,nextState,depth-1) for nextState in listNextStates]
              return sum(listStuff)/len(listStuff)

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction
      
      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    numOfAgent=gameState.getNumAgents();
    trueDepth=numOfAgent*self.depth
    LegalActions=gameState.getLegalActions(0)
    if Directions.STOP in LegalActions: 
        LegalActions.remove(Directions.STOP)
    listNextStates=[gameState.generateSuccessor(0,action) for action in LegalActions ]
    #print(self.Expectimax_Value(numOfAgent,0,gameState,trueDepth))
    v=[self.Expectimax_Value(numOfAgent,1,nextGameState,trueDepth-1) for nextGameState in listNextStates] 
    MaxV=max(v)
    listMax=[]
    for i in range(0,len(v)):
        if v[i]==MaxV:
             listMax.append(i)
    i = random.randint(0,len(listMax)-1)
    
    print(LegalActions)
    print(v)
    print(listMax)
    action=LegalActions[listMax[i]]
    return action

def actualGhostDistance(gameState, ghostPosition):
    from game import Directions
    from game import Actions
    visited = {}
    ghostPosition=util.nearestPoint(ghostPosition)
    curState=startPosition = gameState.getPacmanPosition()
    fringe = util.FasterPriorityQueue()
    Hvalue=util.manhattanDistance(startPosition,ghostPosition)
    curDist=0;
    priorityVal=Hvalue+curDist
    fringe.push(tuple((startPosition,curDist)), priorityVal)
    visited[startPosition] = True
    walls    = gameState.getWalls()
    foodGrid = gameState.getFood()
    isFood   = lambda(x, y): foodGrid[x][y]
    #isGhost  = lambda(x, y): (x, y) in ghostPositions
    while not fringe.isEmpty():
        curState,curDist = fringe.pop()
        # if goal state is found return the distance
        if (curState==ghostPosition):
            #print "returned: %d" % curDist
            return (curState, curDist)
            break
        # if you find a ghost before you find your closest food!! you are screwed =(
        #if (isGhost(curState)):
        #    ghostInRange = True
      
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = curState
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            nextState = (nextx, nexty)
            if ((not walls[nextx][nexty]) and (nextState not in visited)):
                newcurDist = curDist + 1
                priorityVal=util.manhattanDistance(curState,ghostPosition)+newcurDist
                visited[nextState] = True
                fringe.push(tuple((nextState,newcurDist)), priorityVal)    
    return (curState,curDist)
def actualFoodDistance(gameState, targetFood):
    from game import Directions
    from game import Actions
    visited = {}
    curState=startPosition = gameState.getPacmanPosition()
    fringe = util.FasterPriorityQueue()
    Hvalue=util.manhattanDistance(startPosition,targetFood)
    curDist=0;
    priorityVal=Hvalue+curDist
    fringe.push(tuple((startPosition,curDist)), priorityVal)
    visited[startPosition] = True
    walls    = gameState.getWalls()
    foodGrid = gameState.getFood()
    isFood   = lambda(x, y): foodGrid[x][y]
    #isGhost  = lambda(x, y): (x, y) in ghostPositions
    while not fringe.isEmpty():
        curState,curDist = fringe.pop()
        # if goal state is found return the distance
        if (curState==targetFood):
            #print "returned: %d" % curDist
            return (curState, curDist)
            break
        # if you find a ghost before you find your closest food!! you are screwed =(
        #if (isGhost(curState)):
        #    ghostInRange = True
      
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = curState
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            nextState = (nextx, nexty)
            if ((not walls[nextx][nexty]) and (nextState not in visited)):
                
                newcurDist = curDist + 1
                priorityVal=util.manhattanDistance(curState,targetFood)+newcurDist
                visited[nextState] = True
                fringe.push(tuple((nextState,newcurDist)), priorityVal)    
    return (curState,curDist)

def betterEvaluationFunction(currentGameState):
      """
        Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
        evaluation function (question 5).
    
        DESCRIPTION: <write something here so we know what you did>
      """
      if (betterEvaluationFunction.firstCalled): 
          #This indicates the function whether first called or not
          #use this to initialize any variable which you wish don't do this again and again
          print "betterEvaluationFunction is at first Called"
      if currentGameState.isLose():
          return -1e308
      if currentGameState.isWin() : 
          return 1e308          
      returnScore= 0.0
      newPos = currentGameState.getPacmanState().getPosition()
      GhostStates = currentGameState.getGhostStates()
      GhostStates.sort(lambda x,y: disCmp(x.getPosition(),y.getPosition(),newPos))
      GhostPositions = [Ghost.getPosition() for Ghost in GhostStates]
      newScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
      closestGhost=GhostStates[0]
     # closestGhostDistance=util.manhattanDistance(GhostStates[0].getPosition(), newPos)
      #adG=actualGhostDistance(currentGameState,closestGhost.getPosition())
      #capsules = currentGameState.getCapsules()
      #minPill = capsules[0]
      #minDistPill = util.manhattanDistance(minPill, newPos)
      #for Pill in capsules:
      #   curDist = util.manhattanDistance(Pill, newPos)
      #   if curDist==1 : 
       #      minPill=Pill
       #      minDistPill=curDist
       #      break
       #  if (curDist < minDistPill):
       #      minDistPill = curDist
       #      minPill = Pill
   #   print "%d" % len(capsules)
      FoodList = currentGameState.getFood().asList()
     # minPos=min(FoodList,lambda x,y: util.manhattanDistance(newPos,x)-util.manhattanDistance(newPos,y))
      minPos = FoodList[0]
      minDist = util.manhattanDistance(minPos, newPos)
      for food in FoodList:
         curDist = util.manhattanDistance(food, newPos)
         if curDist==1 : 
             minPos=food
             minDist=curDist
             break
         if (curDist < minDist):
             minDist = curDist
             minPos = food
      
     # actualGhostDists = actualGhostDistance(currentGameState, GhostPositions)
      targetFoodPosition, closestFoodDistance = actualFoodDistance(currentGameState, minPos)
     # print(targetFoodPosition,closestFoodDistance)
      # for any centers of mass created by any two ghosts will be recorded
      # the closest one from Pacman will be noted and a special weight will be assigned.
      #allTwoGhosts = allCombo(list(GhostPositions))
      #centerOfGhosts = []
      #for pair in allTwoGhosts:
      #    g1, g2 = pair
      #    x = (g1[0] + g2[0])/2
      #    y = (g1[1] + g2[1])/2
      #    centerOfGhosts = centerOfGhosts + [(x,y)]
      #centerDistOfGhosts = [util.manhattanDistance(center, newPos) for center in centerOfGhosts]
      #fearful = min(centerDistOfGhosts)
      
      # all ghost distances from Pacman
      
      #allRealGhostsDistance = [actualGhostDistance(currentGameState,Pos) for Pos in GhostPositions]#return ghost and its distance
      #allDistGhosts=[Ghost[1] for Ghost in allRealGhostsDistance]
      #closestGhostDistance=min(allDistGhosts)
      #IndexClosestGhost=allDistGhosts.index(closestGhostDistance)
      closestScaredGhostDist=1e308
      closestScaredGhost=None
      closestNormalGhostDist=1e308
      closestNormalGhost=None
      allScaredGhost=[Ghost for Ghost in GhostStates if Ghost.scaredTimer>0]
      allRealScaredGhostDistance=[actualGhostDistance(currentGameState,Pos) for Pos in [ScaredGhost.getPosition() for ScaredGhost in allScaredGhost]]
      allDistScaredGhosts=[Ghost[1] for Ghost in allRealScaredGhostDistance]
      if len(allDistScaredGhosts)!=0:
          closestScaredGhostDist=min(allDistScaredGhosts)
          closestScaredGhost=allScaredGhost[allDistScaredGhosts.index(closestScaredGhostDist)]
      
      allNormalGhost=[Ghost for Ghost in GhostStates if Ghost.scaredTimer<=0]
      allRealNormalGhostDistance=[actualGhostDistance(currentGameState,Pos) for Pos in [NormalGhost.getPosition() for NormalGhost in allNormalGhost]]
      allDistNormalGhosts=[Ghost[1] for Ghost in allRealNormalGhostDistance]
      if len(allDistNormalGhosts)!=0:
          closestNormalGhostDist=min(allDistNormalGhosts)
          closestNormalGhost=allNormalGhost[allDistNormalGhosts.index(closestNormalGhostDist)]
      
      
#      print("all Scared Ghost")
#      print(allScaredGhost)
#      print("Index Scared Ghost")
#      print(IndexesScaredGhost)
#      print("distancesToScaredGhost")
#      print(distancesToScaredGhost)
      wFood, wGhost, wScaredGhost       = [2.0, -6.0, 4.0];
      if (closestNormalGhostDist==0):
          return -1e308
      if (closestScaredGhostDist==0):
          closestScaredGhostDist=0.1
      if (closestNormalGhostDist > 2):
        if closestScaredGhost!=None:
          if (closestScaredGhostDist<closestScaredGhost.scaredTimer):
              wFood, wGhost, wScaredGhost= [0.0, -0.0, 100];
          else:
               wFood, wGhost, wScaredGhost = [4.0, -0.0, 0.0];
        else :
               wFood, wGhost, wScaredGhost = [4.0, -0.0, 0.0];     
      else:
               wFood, wGhost, wScaredGhost= [1, -4, 0];
               
            
      #if (closestGhostDistance > 3):#Ghost too far ignore Ghost
     # if (closestGhostDistance > 2 & minDistPill>2):
         #print("ghostToofar")
      #   if (closestGhost.scaredTimer > closestGhostDistance):
      #      wFood, wGhost, wScaredGhost,wPill = [2.0, -0.0, 8.0,0];
       #  else:
       #     wFood, wGhost, wScaredGhost,wPill = [4.0, -0.0, 0.0,0];
      #else:
      #    if (minDistPill<2 & closestGhostDistance > 2):
      #      wFood, wGhost, wScaredGhost,wPill = [2.0, -0.0, 8.0,-10];
      #if (closestGhostDistance<=2):
      #  if (closestGhost.scaredTimer >0):
      #      wFood, wGhost, wScaredGhost,wPill = [2.0, -0.0, 8.0,-10];
      #  else:
       #     wFood, wGhost, wScaredGhost,wPill = [4.0, -0.0, 0.0,100];
      #if (ghostInRange and closestFoodDistance < 5): 
      #   wFood, wGhost, wScaredGhost    = [2.0, -5.0, 4.0];
      # you are gonna die anyway, why not die fat
      #if (fearful < 3 and crisis < (currentGameState.getNumAgents()-1)*3):
      #   wFood, wGhost, wScaredGhost    = [9.0, -7.0, 9.0];   
      #if (closestGhost.scaredTimer > 3):
         # returnScore = (wFood/closestFoodDistance+wScaredGhost/closestGhostDistance)+wPill/minDistPill+currentGameState.getScore()
      #else: 
      returnScore=(wFood/(closestFoodDistance)+(wGhost)/closestNormalGhostDist+(wScaredGhost)/(closestScaredGhostDist))+currentGameState.getScore()
#      print "-"*30
#      print "all Ghost Distance "
#      print(currentGameState) 
#      print allDistGhosts
#      print "All Ghost position"
#      print (GhostPositions)
#      print ("All Ghosts:")
#      print (GhostStates)
#      print ("All Ghosts Distance")
#      print (allRealGhostsDistance)
#      print "Closest Ghost Distance %d" %  closestGhostDistance
#      print("Return Score:")
#      print(returnScore)
      betterEvaluationFunction.firstCalled=False;
      return returnScore

def allCombo(myList):
    all = []
    if len(myList) == 0:
        return []
    head = myList.pop()
    for i in myList:
        all = all + [(head, i)]
    return all + allCombo(myList)

DISTANCE_CALCULATORS = {}