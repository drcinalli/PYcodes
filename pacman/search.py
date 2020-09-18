"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def getStartState(self):
     """
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 74].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.18].
  """
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()
  
  #seta inicio das variaveis
  no_puro = []
  #coloca tupla: primeira posicao/acao/custo
  # a primeira posicao vem da classe abstrata q eh reflexo da classe dentro de SearchAgents.py
  no_puro.append((problem.getStartState(),'',0)) 
  
  no_visitado = []
       
  #chama funcao recursiva de busca em profundidade     
  return DFSRecursive(no_puro, problem, no_visitado)
    
def DFSRecursive(no_puro, problem, no_visitado):
    no = no_puro.pop() #tira do topo da pilha    
    no_visitado.append(no[0]) #ja coloca no no visitado
    #print no[0]
    
    #se realmente eh a posicao GOAL = volta
    if problem.isGoalState(no[0]):
        return []
    #se nao for a posicao final, le seus sucessores e passa recursivamente 
    #nesse caso o proprio FOR pega automaticamente o no mais a esquerda 
    for i in problem.getSuccessors(no[0]):
        
        #verifica repeticao de no ... no caso ele nao pega no repetido        
        if i[0] not in no_visitado: # se o no (dentro da tupla) nao foi visitado
            no_puro.append(i) #pega o no pela tupla: sucessor/acao pra chegar nele/custo pra chegar nele
            
            ListaAction = DFSRecursive(no_puro, problem, no_visitado)
            #aqui eh pra pegar a acao  q levou ateh o caminho final
            # qdo entra aqui todos os returns daqui pra tras vao receber essa acao
            #dessa forma ele mantem o historico.
            if (ListaAction != None):
                ListaAction.insert(0,i[1])
                #print i[1]                
                return ListaAction


def depthFirstSearchRep(problem):
  """
  Search the deepest nodes in the search tree first [p 74].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.18].
  """
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()
  
  #seta inicio das variaveis
  no_puro = []
  #coloca tupla: primeira posicao/acao/custo
  # a primeira posicao vem da classe abstrata q eh reflexo da classe dentro de SearchAgents.py
  no_puro.append((problem.getStartState(),'',0)) 
  
  no_visitado = []
       
  #chama funcao recursiva de busca em profundidade     
  return DFSRecursiveREP(no_puro, problem, no_visitado)
    
def DFSRecursiveREP(no_puro, problem, no_visitado):
    no = no_puro.pop() #tira do topo da pilha    
    no_visitado.append(no[0]) #ja coloca no no visitado
    
    #se realmente eh a posicao GOAL = volta
    if problem.isGoalState(no[0]):
        return []
    #se nao for a posicao final, le seus sucessores e passa recursivamente 
    #nesse caso o proprio FOR pega automaticamente o no mais a esquerda 
    for i in problem.getSuccessors(no[0]):
        
        #verifica repeticao de no ... no caso ele nao pega no repetido        
        #if i[0] not in no_visitado: # se o no (dentro da tupla) nao foi visitado
            no_puro.append(i) #pega o no pela tupla: sucessor/acao pra chegar nele/custo pra chegar nele
            
            ListaAction = DFSRecursiveREP(no_puro, problem, no_visitado)
            #aqui eh pra pegar a acao  q levou ateh o caminho final
            # qdo entra aqui todos os returns daqui pra tras vao receber essa acao
            #dessa forma ele mantem o historico.
            if (ListaAction != None):
                ListaAction.insert(0,i[1])
                return ListaAction


def depthFirstSearchLimitado(problem):
  """
  Search the deepest nodes in the search tree first [p 74].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.18].
  """
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()
  
  #seta inicio das variaveis
  no_puro = []
  #coloca tupla: primeira posicao/acao/custo
  # a primeira posicao vem da classe abstrata q eh reflexo da classe dentro de SearchAgents.py
  no_puro.append((problem.getStartState(),'',0)) 
  
  no_visitado = 0
       
  #chama funcao recursiva de busca em profundidade     
  return DFSRecursiveLimitado(no_puro, problem, no_visitado)
    
def DFSRecursiveLimitado(no_puro, problem, no_visitado):
    no = no_puro.pop() #tira do topo da pilha    
    #no_visitado.append(no[0]) #ja coloca no no visitado
    no_visitado = no_visitado +1
    #print no_visitado
    
    #se realmente eh a posicao GOAL = volta
    if problem.isGoalState(no[0]):
        return []
    #pergunto se o limite chegou, se chegou e mando voltar com vazio e menos 1?
    if (no_visitado < 20): 
      
      #se nao for a posicao final, le seus sucessores e passa recursivamente 
      #nesse caso o proprio FOR pega automaticamente o no mais a esquerda 
      for i in problem.getSuccessors(no[0]):
        #print i[0]
          #verifica repeticao de no ... no caso ele nao pega no repetido        
          #if i[0] not in no_visitado: # se o no (dentro da tupla) nao foi visitado
        no_puro.append(i) #pega o no pela tupla: sucessor/acao pra chegar nele/custo pra chegar nele
        
        ListaAction = DFSRecursiveLimitado(no_puro, problem, no_visitado)
        #aqui eh pra pegar a acao  q levou ateh o caminho final
        # qdo entra aqui todos os returns daqui pra tras vao receber essa acao
        #dessa forma ele mantem o historico.
        if (ListaAction != None):
            ListaAction.insert(0,i[1])
            return ListaAction

    
    #no_visitado = no_visitado -1
      
    
     

def breadthFirstSearchRecursive(problem):
  "Search the shallowest nodes in the search tree first. [p 74]"
  "*** YOUR CODE HERE ***"
 #util.raiseNotDefined()
  
  #seta inicio das variaveis
  no_puro = util.Queue()
  
  #coloca tupla: primeira posicao/acao/custo
  # a primeira posicao vem da classe abstrata q eh reflexo da classe dentro de SearchAgents.py
  no_puro.push((problem.getStartState(),'',0)) 
  
  no_visitado = []
       
  #chama funcao recursiva de busca em amplitude     
  return BFSRecursive(no_puro, problem, no_visitado)
    
def BFSRecursive(no_puro, problem, no_visitado):

  no_filhos = util.Queue()
  #pega todos os nos do nivel.
  while not no_puro.isEmpty():
    #pega elemento
    no = no_puro.pop()
    no_visitado.append(no[0]) #ja coloca NO no visitado
    print no[0]
    
    #vejo se chegou onde eu queria
    #se realmente eh a posicao GOAL = volta
    if problem.isGoalState(no[0]):
        return no[0] #se chegou mando voltar
        
    #enfileiro cada um dos seus sucessores
    for i in problem.getSuccessors(no[0]):
            
            #verifica repeticao de no ... no caso ele nao pega no repetido        
            if i[0] not in no_visitado: # se o no (dentro da tupla) nao foi visitado
                no_filhos.push(i) #pega o no pela tupla: sucessor/acao pra chegar nele/custo pra chegar nele
                #print i[0] 

    # Aqui eu tenho todos os sucessores do nivel abaixo (filhos) na mao
    # Entao eu mando recursivamente pra funcao ver se um deles eh o GOAL
    # e de novo procurar o nivel de baixo.
    BFSRecursive(no_filhos, problem, no_visitado)
    
   


def breadthFirstSearch(problem):
    "Search the shallowest nodes in the search tree first. [p 81]"
    "*** YOUR CODE HERE ***"
    
    #estrutura para armazenar o caminho do algoritmo, sera usada pra saber 
    #voltar do GOAL ate o root
    #ele guarda o caminho e a acao do no anterior ate o no corrente
    #nesse caso ele esta guardando no array[getstartstate] o valor none, none
    #eh um array com uma estrutura dentro na verdade 
    caminho = { problem.getStartState():(None,None) }
    
    visitado=[]
              
    bfsQ = util.Queue()
    bfsQ.push( problem.getStartState() ) #colocar primeira posicao
    
    while not bfsQ.isEmpty() :
        V = bfsQ.pop() #pego
        visitado.append(V) #visito
        
        if problem.isGoalState(V):   #e o q quero?
            resultado=[]
            aux=V #copia para mudancas de indices
            
            # ##### logica para fazer o trace de actions pro jogo
            while caminho[aux][0]!=None: #faz ate chegar no root do jogo pq soh no root o pai eh vazio
                resultado.append(caminho[aux][1]) #atribui acao
                aux=caminho[aux][0]      #pega o pai
            resultado.reverse()
            return resultado
          
        #para cada sucessor de V....  
        for W in problem.getSuccessors(V):
            #vejo se houve visita no no
            if W not in visitado :
                #se nao houve visita eu coloco no array W, seu pai e seu action
                #ou seja, seu caminho ateh ele.
                if W[0] not in caminho :
                  caminho[W[0]] = ( V,W[1] ) #pai e action
                  bfsQ.push(W[0]) #jogo pra Fila
    return 

def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()
  #estrutura para armazenar o caminho do algoritmo, sera usada pra saber 
  #voltar do GOAL ate o root
  #ele guarda o caminho e a acao do no anterior ate o no corrente
  #nesse caso ele esta guardando no array[getstartstate] o valor none, none
  #eh um array com uma estrutura dentro na verdade 
  caminho = { problem.getStartState():(None,None) }
  prioridade = 1
  visitado=[]
            
  bfsQ = util.PriorityQueue()
  bfsQ.push( problem.getStartState(), 1 ) #colocar primeira posicao + Custo 
  
  while not bfsQ.isEmpty() :
      V = bfsQ.pop() #pego
      
      # #####################################
      # aqui eh a mudanca em relacao ao BFS #
      # aqui soh serve pra setar um custo de#
      # prioridade
      # #####################################
  
      # nao poderia simplesmente pegar o custo de um dos filhos aqui????
      # ja q tudo tem custo 1 no pacman
      # ou melhor... ir somando 1 ou entao o proprio custo.

      # sim ... funcionou
      prioridade = prioridade + 1      
      # Nao funciona bem .... pq toda vez q eu dava um pop eu somava 1 na prioridade
      # e qdo eu ia pro outro lado da arvore o valor estava errado.
      # basta desenhar e fica facil de ver.
      
      # bem ... esse aqui tbem nao faz mto bem essa parte... o problema eh o pacman ser
      # todo de custo 1

      #tmp2 = problem.getSuccessors(V)
      #tmp =  problem.getSuccessors(tmp2[0][0])
      #state = []
      #for state in tmp:
      #  if state[0] == V:
      #    prioridade = prioridade + state[2]      
      
      
      visitado.append(V) #visito
     
      # Atencao Daqui pra baixo nao muda nada em relacao ao BFS
      #       
      if problem.isGoalState(V):   #e o q quero?
        
          resultado=[]
          aux=V #copia para mudancas de indices
          
          # ##### logica para fazer o trace de actions pro jogo
          while caminho[aux][0]!=None: #faz ate chegar no root do jogo pq soh no root o pai eh vazio
              resultado.append(caminho[aux][1]) #atribui acao
              aux=caminho[aux][0]      #pega o pai
          resultado.reverse()
          return resultado
        
      #para cada sucessor de V....  
      for W in problem.getSuccessors(V):
          #vejo se houve visita no no
          if W not in visitado :
              #se nao houve visita eu coloco no array W, seu pai e seu action
              #ou seja, seu caminho ateh ele.
              if W[0] not in caminho :
                caminho[W[0]] = ( V,W[1] ) #pai e action
                # aqui eu somo a prioridade no custo pois no PACman o custo eh sempre 1
                bfsQ.push(W[0],W[2]+prioridade) #jogo pra Fila
  return 


def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  "*** YOUR CODE HERE ***"
  #util.raiseNotDefined()

  #estrutura para armazenar o caminho do algoritmo, sera usada pra saber 
  #voltar do GOAL ate o root
  #ele guarda o caminho e a acao do no anterior ate o no corrente
  #nesse caso ele esta guardando no array[getstartstate] o valor none, none
  #eh um array com uma estrutura dentro na verdade 
  caminho = { problem.getStartState():(None,None) }
  prioridade = 1
  visitado=[]
            
  bfsQ = util.PriorityQueue()
  bfsQ.push( problem.getStartState(), heuristic(problem.getStartState(),problem) ) #colocar primeira posicao + Custo 
  
  while not bfsQ.isEmpty() :
      V = bfsQ.pop() #pego
      
      # #####################################
      # aqui eh a mudanca em relacao ao BFS #
      # aqui soh serve pra setar um custo de#
      # prioridade
      # #####################################
  
      # nao poderia simplesmente pegar o custo de um dos filhos aqui????
      # ja q tudo tem custo 1 no pacman
      # ou melhor... ir somando 1 ou entao o proprio custo.

      # sim ... funcionou
      prioridade = prioridade + 1      
      # Nao funciona bem .... pq toda vez q eu dava um pop eu somava 1 na prioridade
      # e qdo eu ia pro outro lado da arvore o valor estava errado.
      # basta desenhar e fica facil de ver.
      
      # bem ... esse aqui tbem nao faz mto bem essa parte... o problema eh o pacman ser
      # todo de custo 1

      #tmp2 = problem.getSuccessors(V)
      #tmp =  problem.getSuccessors(tmp2[0][0])
      #state = []
      #for state in tmp:
      #  if state[0] == V:
      #    prioridade = prioridade + state[2]      
      
      
      visitado.append(V) #visito
     
      # Atencao Daqui pra baixo nao muda nada em relacao ao BFS
      #       
      if problem.isGoalState(V):   #e o q quero?
        
          resultado=[]
          aux=V #copia para mudancas de indices
          
          # ##### logica para fazer o trace de actions pro jogo
          while caminho[aux][0]!=None: #faz ate chegar no root do jogo pq soh no root o pai eh vazio
              resultado.append(caminho[aux][1]) #atribui acao
              aux=caminho[aux][0]      #pega o pai
          resultado.reverse()
          return resultado
        
      #para cada sucessor de V....  
      for W in problem.getSuccessors(V):
          #vejo se houve visita no no
          if W not in visitado :
              #se nao houve visita eu coloco no array W, seu pai e seu action
              #ou seja, seu caminho ateh ele.
              if W[0] not in caminho :
                caminho[W[0]] = ( V,W[1] ) #pai e action
                # aqui eu somo a prioridade no custo pois no PACman o custo eh sempre 1 + a heuristica da distancia entre o no e o goal
                bfsQ.push(W[0],W[2]+prioridade+heuristic(W[0],problem)) #jogo pra Fila
  return 

     
def bestFirstSearch(problem, heuristic=nullHeuristic):
  #util.raiseNotDefined()

  #estrutura para armazenar o caminho do algoritmo, sera usada pra saber 
  #voltar do GOAL ate o root
  #ele guarda o caminho e a acao do no anterior ate o no corrente
  #nesse caso ele esta guardando no array[getstartstate] o valor none, none
  #eh um array com uma estrutura dentro na verdade 
  caminho = { problem.getStartState():(None,None) }
  prioridade = 1
  visitado=[]
            
  bfsQ = util.PriorityQueue()
  bfsQ.push( problem.getStartState(), heuristic(problem.getStartState(),problem) ) #colocar primeira posicao + Custo 
  
  while not bfsQ.isEmpty() :
      V = bfsQ.pop() #pego
      
      # #####################################
      # aqui eh a mudanca em relacao ao BFS #
      # aqui soh serve pra setar um custo de#
      # prioridade
      # #####################################
  
      # nao poderia simplesmente pegar o custo de um dos filhos aqui????
      # ja q tudo tem custo 1 no pacman
      # ou melhor... ir somando 1 ou entao o proprio custo.

      # sim ... funcionou
      prioridade = prioridade + 1      
      # Nao funciona bem .... pq toda vez q eu dava um pop eu somava 1 na prioridade
      # e qdo eu ia pro outro lado da arvore o valor estava errado.
      # basta desenhar e fica facil de ver.
      
      # bem ... esse aqui tbem nao faz mto bem essa parte... o problema eh o pacman ser
      # todo de custo 1

      #tmp2 = problem.getSuccessors(V)
      #tmp =  problem.getSuccessors(tmp2[0][0])
      #state = []
      #for state in tmp:
      #  if state[0] == V:
      #    prioridade = prioridade + state[2]      
      
      
      visitado.append(V) #visito
     
      # Atencao Daqui pra baixo nao muda nada em relacao ao BFS
      #       
      if problem.isGoalState(V):   #e o q quero?
        
          resultado=[]
          aux=V #copia para mudancas de indices
          
          # ##### logica para fazer o trace de actions pro jogo
          while caminho[aux][0]!=None: #faz ate chegar no root do jogo pq soh no root o pai eh vazio
              resultado.append(caminho[aux][1]) #atribui acao
              aux=caminho[aux][0]      #pega o pai
          resultado.reverse()
          return resultado
        
      #para cada sucessor de V....  
      for W in problem.getSuccessors(V):
          #vejo se houve visita no no
          if W not in visitado :
              #se nao houve visita eu coloco no array W, seu pai e seu action
              #ou seja, seu caminho ateh ele.
              if W[0] not in caminho :
                caminho[W[0]] = ( V,W[1] ) #pai e action
                # aqui eu somo apenas a heuristica da distancia entre o no e o goal
                bfsQ.push(W[0],heuristic(W[0],problem)) #jogo pra Fila
  return 
   

def hillClimbingSearch(problem, heuristic=nullHeuristic):
  #util.raiseNotDefined()

  #estrutura para armazenar o caminho do algoritmo, sera usada pra saber 
  #voltar do GOAL ate o root
  #ele guarda o caminho e a acao do no anterior ate o no corrente
  #nesse caso ele esta guardando no array[getstartstate] o valor none, none
  #eh um array com uma estrutura dentro na verdade 
  caminho = { problem.getStartState():(None,None) }
  prioridade = 1
  visitado=[]
            
  bfsQ = util.PriorityQueue()
  bfsQ.push( problem.getStartState(), heuristic(problem.getStartState(),problem) ) #colocar primeira posicao + Custo 
  
  while not bfsQ.isEmpty() :
      V = bfsQ.pop() #pego
      
      # #####################################
      # aqui eh a mudanca em relacao ao BFS #
      # aqui soh serve pra setar um custo de#
      # prioridade
      # #####################################
  
      # nao poderia simplesmente pegar o custo de um dos filhos aqui????
      # ja q tudo tem custo 1 no pacman
      # ou melhor... ir somando 1 ou entao o proprio custo.

      # sim ... funcionou
      prioridade = prioridade + 1      
      # Nao funciona bem .... pq toda vez q eu dava um pop eu somava 1 na prioridade
      # e qdo eu ia pro outro lado da arvore o valor estava errado.
      # basta desenhar e fica facil de ver.
      
      # bem ... esse aqui tbem nao faz mto bem essa parte... o problema eh o pacman ser
      # todo de custo 1

      #tmp2 = problem.getSuccessors(V)
      #tmp =  problem.getSuccessors(tmp2[0][0])
      #state = []
      #for state in tmp:
      #  if state[0] == V:
      #    prioridade = prioridade + state[2]      
      
      
      visitado.append(V) #visito
     
      # Atencao Daqui pra baixo nao muda nada em relacao ao BFS
      #       
      if problem.isGoalState(V):   #e o q quero?
        
          resultado=[]
          aux=V #copia para mudancas de indices
          
          # ##### logica para fazer o trace de actions pro jogo
          while caminho[aux][0]!=None: #faz ate chegar no root do jogo pq soh no root o pai eh vazio
              resultado.append(caminho[aux][1]) #atribui acao
              aux=caminho[aux][0]      #pega o pai
          resultado.reverse()
          return resultado
        
      #aqui eu zero a fila, pois no HillClimbing nao eh preciso guardar o historico
      while not bfsQ.isEmpty() :
        bfsQ.pop()
        
      #para cada sucessor de V....  
      for W in problem.getSuccessors(V):
          #vejo se houve visita no no
          if W not in visitado :
              #se nao houve visita eu coloco no array W, seu pai e seu action
              #ou seja, seu caminho ateh ele.
              if W[0] not in caminho :
                caminho[W[0]] = ( V,W[1] ) #pai e action
                # aqui eu somo apenas a heuristica da distancia entre o no e o goal
                bfsQ.push(W[0],heuristic(W[0],problem)) #jogo pra Fila
  return 

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch


