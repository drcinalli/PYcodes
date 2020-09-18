# keyboardAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import Agent
from game import Directions
from util import manhattanDistance

import random, util

		
	
class KeyboardAgent(Agent):
	"""
	An agent controlled by the keyboard.
	"""
	# NOTE: Arrow keys also work.
	WEST_KEY	= 'a' 
	EAST_KEY	= 'd' 
	NORTH_KEY = 'w' 
	SOUTH_KEY = 's'
	STOP_KEY = 'q'

	def __init__( self, index = 0 ):
	
		self.lastMove = Directions.STOP
		self.index = index
		self.keys = []
	
	

	def Bussola( self, pacman, alvo, legal, currentGameState):
		
		dir='STOP'
		dist=util.manhattanDistance(pacman, alvo)
		
		for move in legal:
			successorGameState = currentGameState.generatePacmanSuccessor(move)
			newPos = successorGameState.getPacmanState().getPosition()
			
			aux=util.manhattanDistance(newPos, alvo)
			if aux < dist:
				dir = move
				dist = aux
		return dir

	def pegaNumero( self, move):
		
		
		if move == 'STOP' or move == 'Stop' : return 0
		elif move == 'North': return 1
		elif move == 'South': return 2
		elif move == 'East': return 3
		elif move == 'West': return 4
		
	
 
    
	    
	def disCmp(self,x,y,newPos):
	    if (util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))<0: return -1
	    else: 
	        if (util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))>0: return 1
	        else:
	            return 0
	
		
			
	def writeTraining(self, legal, currentGameState):
		for move in legal:
			print move
			successorGameState = currentGameState.generatePacmanSuccessor(move)
			newPos = successorGameState.getPacmanPosition()
			#Pos=currentGameState.getPacmanPosition()
			oldFood = currentGameState.getFood()
			newGhostStates = successorGameState.getGhostStates()
			newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
			
			
			
			#descobrir a distancia da comida mais proxima
			ListComida=oldFood.asList()
			ListComida.sort(lambda x,y: util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))
			DistanciaComida=util.manhattanDistance(newPos, ListComida[0])
			
			#setar valor das distancias do fantasma
			PosicaoFantasma=[Ghost.getPosition() for Ghost in newGhostStates]
			#PosicaoFantasma.sort(lambda x,y: self.disCmp(x,y,Pos))
			DistanciaFantasma1 = int(util.manhattanDistance(newPos, PosicaoFantasma[0]))
			DistanciaFantasma2 = int(util.manhattanDistance(newPos, PosicaoFantasma[1]))
			
			#pega o medo
			fantasma_medo = min(newScaredTimes)			

			varFile = open("treinamento.txt","a")
			
			input = []
			
			
			pesosFantasmas = []
			vivo = []
			vivo.append(1)
			
			if(DistanciaFantasma1 <= DistanciaFantasma2):
				dir_fantasma=self.Bussola(currentGameState.getPacmanPosition() ,PosicaoFantasma[0],legal,currentGameState)
				distancia = DistanciaFantasma1
				posicao_fantasma_prox=PosicaoFantasma[0]
			else:
				dir_fantasma=self.Bussola(currentGameState.getPacmanPosition() ,PosicaoFantasma[1],legal,currentGameState)
				distancia = DistanciaFantasma2
				posicao_fantasma_prox=PosicaoFantasma[1]
				
			#dir_fantasma2=self.Bussola(currentGameState.getPacmanPosition() ,ghostPositions[1],legal,currentGameState)
			
			#dir_fantasma_perto = min(dir_fantasma1,dir_fantasma2)
			
			input.append(distancia)
			input.append(fantasma_medo)
			input.append(int(DistanciaComida))
			x_fantasma=self.pegaNumero(dir_fantasma)
			#input.append(x_fantasma)
			y_pacman=int(self.pegaNumero(move))	
			#input.append(y_pacman)
			
			#se perto			
			if distancia <3:
				if (x_fantasma == y_pacman and fantasma_medo > 4):
					vivo[0]=1
				elif (x_fantasma == y_pacman and fantasma_medo <= 4):	 
					vivo[0]=0
				#se as duas direcoes diferentes
				elif (x_fantasma != y_pacman and fantasma_medo > 4):
					#vejo se a dist. encurta ou se afasta
					if util.manhattanDistance(newPos, posicao_fantasma_prox) < util.manhattanDistance(currentGameState.getPacmanPosition(), posicao_fantasma_prox):
						vivo[0]=1
					else:
						vivo[0]=0	
				elif (x_fantasma != y_pacman and fantasma_medo <= 4):	 
					#vejo se a dist. encurta ou se afasta
					if util.manhattanDistance(newPos, posicao_fantasma_prox) < util.manhattanDistance(currentGameState.getPacmanPosition(), posicao_fantasma_prox):
						vivo[0]=0
					else:
						vivo[0]=1	

			else:
				#vejo se a dist. encurta ou se afasta
				if util.manhattanDistance(newPos, ListComida[0]) < util.manhattanDistance(currentGameState.getPacmanPosition(), ListComida[0]):
					vivo[0]=1
				else:
					vivo[0]=0	
			
				
			input = [input]	
			input.append(vivo)
			
			varFile.write(str(input)+",\n")
			
			varFile.close()
					
				
		
	
	def getAction( self, state):
		from graphicsUtils import keys_waiting
		from graphicsUtils import keys_pressed
		keys = keys_waiting() + keys_pressed()
		if keys != []:
			self.keys = keys
		
		legal = state.getLegalActions(self.index)
		#self.writeTraining(legal, state)
		move = self.getMove(legal)
		self.writeTraining( legal,state)
		
		if move == Directions.STOP:
			# Try to move in the same direction as before
			if self.lastMove in legal:
				move = self.lastMove
		
		if (self.STOP_KEY in self.keys) and Directions.STOP in legal: move = Directions.STOP

		if move not in legal:
			move = random.choice(legal)
			
		self.lastMove = move

				
		return move

	def disCmp(x,y,newPos):
		if (util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))<0: return -1
		else: 
			if (util.manhattanDistance(newPos, x)-util.manhattanDistance(newPos, y))>0: return 1
			else:
				return 0		

	def getMove(self, legal):
		move = Directions.STOP
		if	 (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:	move = Directions.WEST
		if	 (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal: move = Directions.EAST
		if	 (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:	 move = Directions.NORTH
		if	 (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal: move = Directions.SOUTH
		return move