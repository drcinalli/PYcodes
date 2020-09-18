'''
Created on 07/04/2012

@author: quatrosem
'''
# Tic Tac Toex

import operator, sys, random

jX = 'x'
jO = 'o'
contador = 0

def desenhaTabuleiro(tabu):

    print('    |     |')
    print(' ' + tabu[7] + '  |  ' + tabu[8] + '  |  ' + tabu[9])
    print('    |     |')
    print('---------------')
    print('    |     |')
    print(' ' + tabu[4] + '  |  ' + tabu[5] + '  |  ' + tabu[6])
    print('    |     |')
    print('---------------')
    print('    |     |')
    print(' ' + tabu[1] + '  |  ' + tabu[2] + '  |  ' + tabu[3])
    print('    |     |')
    
def desenhaTabuleiroVazio(tabu):

    print('    |     |')
    print(' ' + '7' + '  |  ' + '8' + '  |  ' +  '9')
    print('    |     |')
    print('---------------')
    print('    |     |')
    print(' ' +  '4' + '  |  ' + '5' + '  |  ' + '6')
    print('    |     |')
    print('---------------')
    print('    |     |')
    print(' ' + '1' + '  |  ' + '2' + '  |  ' + '3')
    print('    |     |')

    
def humanoEscolheLetra():
    letra = ''

    while not (letra.upper() == 'X' or letra.upper() == 'O'):
        letra = raw_input("Por Favor, escolha X ou O?")
        #print letra.upper()

    # o primeiro na ordem eh o humano, o segundo sera o computador.
    if letra.upper() == 'X':
        return ['X', 'O']
    else:
        return ['O', 'X']

def humanoEscolheAlgoritmo():
    algo = ''

    while not (algo == '0' or algo == '1'or algo == '2'or algo == '3'):
        print " "
        print "0 - Busca Cega (todas as possibilidades)"
        print "1 - Heuristica H1 (menores possibilidades de vitoria do oponente)"
        print "2 - Heuristica H2 (vitorias menos derrotas)"
        print "3 - Heuristica H3 (meio para extremidades) - sem algoritmo MiniMax"
        algo = raw_input("Por Favor, escolha um algoritmo? ")
        #print letra.upper()

    return int(algo)

def humanoEscolheProfundidade(algo):
    prof = "0"
    
    if (algo == 1 or algo == 2):
        while not (int(prof) > 0 and int(prof) < 10):
            print " "            
            prof = raw_input("Escolha uma profundidade (1-9)? ")
        #print letra.upper()

    return int(prof)


def primeiraJogada():
    global contador
    
    if random.randint(0, 1) == 0:
        
        return 'computador'
    else:
        contador = contador + 1
        return 'humano'


def Move(tabu, letra, move):
    tabu[move] = letra


def Campeao(tabu, letra):

    # todas as combinacoes
    return ((tabu[7] == letra and tabu[8] == letra and tabu[9] == letra) or 
    (tabu[4] == letra and tabu[5] == letra and tabu[6] == letra) or 
    (tabu[1] == letra and tabu[2] == letra and tabu[3] == letra) or 
    (tabu[7] == letra and tabu[4] == letra and tabu[1] == letra) or 
    (tabu[8] == letra and tabu[5] == letra and tabu[2] == letra) or 
    (tabu[9] == letra and tabu[6] == letra and tabu[3] == letra) or 
    (tabu[7] == letra and tabu[5] == letra and tabu[3] == letra) or 
    (tabu[9] == letra and tabu[5] == letra and tabu[1] == letra)) 
        
def countCampeao(tabu, letra):

    # todas as combinacoes
    aux = 0
    if (tabu[7] == letra and tabu[8] == letra and tabu[9] == letra):
        aux += 1

    if (tabu[4] == letra and tabu[5] == letra and tabu[6] == letra):
        aux += 1
         
    if (tabu[1] == letra and tabu[2] == letra and tabu[3] == letra):
        aux += 1
     
    if (tabu[7] == letra and tabu[4] == letra and tabu[1] == letra):
        aux += 1
         
    if (tabu[8] == letra and tabu[5] == letra and tabu[2] == letra):
        aux += 1
         
    if (tabu[9] == letra and tabu[6] == letra and tabu[3] == letra):
        aux += 1
         
    if (tabu[7] == letra and tabu[5] == letra and tabu[3] == letra):
        aux += 1
         
    if (tabu[9] == letra and tabu[5] == letra and tabu[1] == letra): 
        aux += 1

    return aux

    
def espacoVazio(tabuleiro, move):
    return tabuleiro[move] == ' '

def TabuleiroCheio(tabu):
    for i in range(1, 10):
        if espacoVazio(tabu, i):
            return False
    return True

def dupTabuleiro(tabuleiro):

    dupTabu = []
    for i in tabuleiro:
        dupTabu.append(i)
    return dupTabu
    
def Sorteio(tabuleiro, lista):

    pMove = []
    for i in lista:
        if espacoVazio(tabuleiro, i):
            pMove.append(i)
    if len(pMove) != 0:
        return random.choice(pMove)
    else:
        return None

def proxMove(tabuleiro):
    #percorre tabuleiro e verifica as posicoes vazias
    return [(pos+1) for pos in range(9) if tabuleiro[pos+1] == " "]

def humanoMove(tabuleiro):
    
    move = ' '
    while move not in '1 2 3 4 5 6 7 8 9'.split() or not espacoVazio(tabuleiro, int(move)):
        move = raw_input("Por Favor, escolha uma casa?")
        #move = input()
    return int(move)
    
    
def computadorMove(tabuleiro, prof):
    oponente = { computador : humano, humano : computador }
    
    def final():
        if Campeao(tabuleiro, computador):
            return +1
        if Campeao(tabuleiro, humano):
            return -1
        return 0
   
    def finalH1(letra):
        #para essa jogada em questao
            
        aux = 0         
        #duplica tabuleiro para fazer as contas
        dupTabu = dupTabuleiro(tabuleiro)
        for move in proxMove(tabuleiro):
            Move(dupTabu, letra, move)
                
        #agora conta as possibilidades de vitoria do oponente
        aux = countCampeao(dupTabu, letra)
        
            #guardo num array
#            H1.append([move,aux])
            
            
        #agora conta as possibilidades de vitoria do oponente
        #aux = countCampeao(dupTabu, letra)
        
        
        #diminuo um do outro e esse eh o valor retornado
        
        return aux    
            #guardo num array
#            H1.append([move,aux])
        
#        random.shuffle(H1)
#        H1.sort(key = lambda (move, aux): aux)
        #print H1
        #print H1[-1][0]
#        return H1[-1][0]
    
    def finalH2():
        return 0

    def funcaoAvaliacaoCega(move, letra):
        try:
            global contador
            contador = contador + 1
        
            Move(tabuleiro,letra, move)
            if TabuleiroCheio(tabuleiro) or Campeao(tabuleiro, letra):
                return final()
            
            # AQUI. Chama recursivamente para cada opcao desde a jogada atual.
            outcomes = (funcaoAvaliacaoCega(next_move, oponente[letra]) for next_move in proxMove(tabuleiro))
            
            #se for o computador, Minimiza as chances 
            # AQUI o Min e o Max parecem trocados, mas eh isso mesmo.
            #Basta imaginar q no algoritimo e na representacao do livro
            #o ultimo no da arvore ah a jogada do adversario e qdo ele volta um no eh a vez do computador, mas o valor 
            # eh o minimo
            if letra == computador:
                #return min(outcomes)
                min_element = 1
                for o in outcomes:
                    if o == -1:
                        return o
                    min_element = min(o,min_element)
                return min_element
            else:
                #return max(outcomes)                
                max_element = -1
                for o in outcomes:
                    if o == +1:
                        return o
                    max_element = max(o,max_element)
                return max_element
                        
        finally:
            #print "aaaaa " + str(move)
            #desenhaTabuleiro(tabuleiro)
            Move(tabuleiro, ' ',move)  
            #desenhaTabuleiro(tabuleiro)     

    def funcaoAvaliacaoH1(move, letra, profundidade_entrada):
        try:
            global contador
            contador = contador + 1
            profundidade_entrada += 1
            
            
            Move(tabuleiro,letra, move)
            if TabuleiroCheio(tabuleiro) or Campeao(tabuleiro, letra) or profundidade_entrada>=prof:
                #print letra + " " + str(finalH1(oponente[letra])) 
                return finalH1(oponente[letra])
            
            # AQUI. Chama recursivamente para cada opcao desde a jogada atual.
            outcomes = (funcaoAvaliacaoH1(next_move, oponente[letra],profundidade_entrada) for next_move in proxMove(tabuleiro))
            #print outcomes
            #print str(profundidade_entrada) + "   " + str(prof)
            #se for o computador, Minimiza as chances 
            # AQUI o Min e o Max parecem trocados, mas eh isso mesmo.
            #Basta imaginar q no algoritimo e na representacao do livro
            #o ultimo no da arvore ah a jogada do adversario e qdo ele volta um no eh a vez do computador, mas o valor 
            # eh o minimo
            if letra == computador:
                return min(outcomes)
                #min_element = 1
                #for o in outcomes:
                #    if o == -1:
                #        return o
                #    min_element = min(o,min_element)
                #return min_element
            else:
                return max(outcomes)                
                #max_element = -1
                #for o in outcomes:
                #   if o == +1:
                #        return o
                #    max_element = max(o,max_element)
                #return max_element
                        
        finally:
            #print "aaaaa " + str(move)
            #desenhaTabuleiro(tabuleiro)
            Move(tabuleiro, ' ',move)  
            #desenhaTabuleiro(tabuleiro)     

   
    # #############
    #move = ' '
    #while move not in '1 2 3 4 5 6 7 8 9'.split() or not espacoVazio(tabuleiro, int(move)):
    #    move = raw_input("Por Favor, escolha uma casa?")       
    # #################
    
    #busca CEGA
    if algo == 0:
            
        moves = [(move, funcaoAvaliacaoCega(move, computador)) for move in proxMove(tabuleiro)]
        random.shuffle(moves)
        moves.sort(key = lambda (move, pontos): pontos)
        #print moves
        #print moves[-1][0]
        #board.makeMove(moves[-1][0], player)
        #print "hahahaha " + str( funcaoAvaliacaoCega(int(move), computador))
        #print proxMove(tabuleiro)
        
        return int(moves[-1][0])
    
    #busca Heuristica 1 - menores chances de vitoria do oponente
    elif algo == 1:
        #vejo antes se preciso jogar uma defesa ou um ataque final

 
        moves = [(move, funcaoAvaliacaoH1(move, computador, 0)) for move in proxMove(tabuleiro)]
        random.shuffle(moves)
        moves.sort(key = lambda (move, pontos): pontos)
        print moves
        #print moves
        #print moves[-1][0]
        #board.makeMove(moves[-1][0], player)
        #print "hahahaha " + str( funcaoAvaliacaoCega(int(move), computador))
        #print proxMove(tabuleiro)
        
        return int(moves[-1][0])
    

        
        #ataque final?
#        for i in range(1, 10):
#            dupTabu = dupTabuleiro(tabuleiro)
#            if espacoVazio(dupTabu, i):
#                Move(dupTabu, computador, i)
#                if Campeao(dupTabu, computador):
#                    return i
#        # defesa?
#        for i in range(1, 10):
#            dupTabu = dupTabuleiro(tabuleiro)
#            if espacoVazio(dupTabu, i):
#                Move(dupTabu, humano, i)
#                if Campeao(dupTabu, humano):
#                    return i
        
        #para cada prox opcao de jogada
#        H1 = []
#        for move in proxMove(tabuleiro):
#            #duplica tabuleiro para fazer as contas
#            dupTabu = dupTabuleiro(tabuleiro)
#            
            #realizo o movimento
#            Move(dupTabu, computador, move)

#            global contador
#            contador = contador + 1
        
            #preenche tudo com letra do humano
#            for k in proxMove(dupTabu):
#                Move(dupTabu, humano, k)
                
            #agora conta as possibilidades de vitoria do oponente
#            aux = countCampeao(dupTabu, humano)
            
            #guardo num array
#            H1.append([move,aux])
        
#        random.shuffle(H1)
#        H1.sort(key = lambda (move, aux): aux)
        #print H1
        #print H1[-1][0]
#        return H1[-1][0]
    
    #busca Heuristica 2 - vitorias menos derrotas
    elif algo == 2:
        #vejo antes se preciso jogar uma defesa ou um ataque final

        moves = [(move, funcaoAvaliacaoCega(move, computador)) for move in proxMove(tabuleiro)]
        random.shuffle(moves)
        moves.sort(key = lambda (move, pontos): pontos)
        #print moves
        #print moves[-1][0]
        #board.makeMove(moves[-1][0], player)
        #print "hahahaha " + str( funcaoAvaliacaoCega(int(move), computador))
        #print proxMove(tabuleiro)
        
        return int(moves[-1][0])
    
        
        #vejo total de vitorias do computador
#        H1 = []
#        for move in proxMove(tabuleiro):
            #duplica tabuleiro para fazer as contas
#            dupTabu = dupTabuleiro(tabuleiro)
            
            #realizo o movimento
            #Move(dupTabu, computador, move)
        
            #preenche tudo com letra do computador
#            for k in proxMove(dupTabu):
#                Move(dupTabu, computador, k)
                
            #agora conta as possibilidades de vitoria
#            aux = countCampeao(dupTabu, computador)
            
            #guardo num array
#            H1.append([move,aux])

        #vejo total de vitorias do computador
#        H2=[]
#        for move in proxMove(tabuleiro):
            #duplica tabuleiro para fazer as contas
#            dupTabu = dupTabuleiro(tabuleiro)
            
            #realizo o movimento
            #Move(dupTabu, computador, move)
        
            #preenche tudo com letra do homem
#            for k in proxMove(dupTabu):
#                Move(dupTabu, humano, k)
                
            #agora conta as possibilidades de vitoria
#            aux = countCampeao(dupTabu, humano)
            
            #guarda valor array
#           H2.append([move,aux])
            
#        random.shuffle(H1)
#        H1.sort(key = lambda (move, aux): aux)
        
        #print H1
        #print H1[-1][0]
#        return H1[-1][0]

  
    
    #busca Heuristica 3 - extremidades
    elif algo == 3:
        
        # centro 
        if espacoVazio(tabuleiro, 5):
            return 5
        # extremidades
        move = Sorteio(tabuleiro, [1, 3, 7, 9])
        if move != None:
            return move
        # vaos
        return Sorteio(tabuleiro, [2, 4, 6, 8])

    
        
        
if __name__ == "__main__":
    tabuleiro = [' '] * 10
    humano, computador = humanoEscolheLetra()
    algo = humanoEscolheAlgoritmo()
    prof = humanoEscolheProfundidade(algo)
    
    #print humano
    vez = primeiraJogada()
 
    desenhaTabuleiroVazio(tabuleiro)    
    
    
    altura = 1
    
    while True:
    
        print('Jogador: ' + vez)
        print " .... " + str(contador)
        
        #humano
        if vez == 'humano':
            
            #faz o movimento
            move = humanoMove(tabuleiro)
            Move(tabuleiro, humano, move)
            
            desenhaTabuleiro(tabuleiro)
            
            #verifica se jogo acabou
            if Campeao(tabuleiro, humano):
                #desenhaTabuleiro(tabuleiro)
                print "Parabens! voce venceu."
                break
            else:
                if TabuleiroCheio(tabuleiro):
                    #empate
                    print "Empate!"
                    break
                else:
                    #muda a ordem
                    vez = "computador"
                                
        #computador
        else:
            
            #faz o movimento
            move = computadorMove(tabuleiro, prof)
            Move(tabuleiro, computador, move)
            
            desenhaTabuleiro(tabuleiro)
            
            #verifica se jogo acabou
            if Campeao(tabuleiro, computador):
                #desenhaTabuleiro(tabuleiro)
                print "Perdeu! eu venci."
                break
            else:
                if TabuleiroCheio(tabuleiro):
                    #empate
                    print "Empate!"
                    break
                else:
                    #muda a ordem
                    vez = "humano"
                                
            
            
                        
    
    
   
       