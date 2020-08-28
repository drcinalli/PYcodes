'''
Created on 30/06/2012

@author: quatrosem
'''

import random
import copy
 
    
#################################
#Definicao de Constantes STRINGS#
################################# 
SEGUNDA, TERCA, QUARTA, QUINTA, SEXTA, SABADO, DOMINGO = "SEGUNDA", "TERCA", "QUARTA", "QUINTA", "SEXTA", "SABADO", "DOMINGO"
SEGUNDA_07_as_10, TERCA_07_as_10, QUARTA_07_as_10, QUINTA_07_as_10, SEXTA_07_as_10, SABADO_07_as_10, DOMINGO_07_as_10 = "SEGUNDA_07_as_10", "TERCA_07_as_10", "QUARTA_07_as_10", "QUINTA_07_as_10", "SEXTA_07_as_10", "SABADO_07_as_10", "DOMINGO_07_as_10"
SEGUNDA_10_as_12, TERCA_10_as_12, QUARTA_10_as_12, QUINTA_10_as_12, SEXTA_10_as_12, SABADO_10_as_12, DOMINGO_10_as_12 = "SEGUNDA_10_as_12", "TERCA_10_as_12", "QUARTA_10_as_12", "QUINTA_10_as_12", "SEXTA_10_as_12", "SABADO_10_as_12", "DOMINGO_10_as_12"
inteligencia_artificial, banco_de_dados, calculo, estatistica, programacao, algoritmo, metodologia_cientifica, estrutura_dados, matematica_discreta, fundamentos_software, engenharia_software, sma = "inteligencia_artificial", "banco_de_dados", "calculo", "estatistica", "programacao", "algoritmo", "metodologia_cientifica", "estrutura_dados", "matematica_discreta", "fundamentos_software", "engenharia_software", "sma"

Ana_Cristina, Ferraz, Viviane, Plastino ="Ana_Cristina", "Ferraz", "Viviane", "Plastino"

    
###################################
#Definicao de Constantes NUMERICAS#
###################################
qtde_aulas_semana=2
qtde_disciplinas=6 

 
 
#######################
### Classe para CSP ###
#######################
class csp:
    def __init__(self, professores,disciplinas,dominio,restricoes_aula_dupla,restricoes_aulas,escolha,
               restricaoSEGUNDA_07_as_10,restricaoSEGUNDA_10_as_12,restricaoTERCA_07_as_10,restricaoTERCA_10_as_12,
               restricaoQUARTA_07_as_10,restricaoQUARTA_10_as_12,restricaoQUINTA_07_as_10,restricaoQUINTA_10_as_12,
               restricaoSEXTA_07_as_10,restricaoSEXTA_10_as_12,restricaoSABADO_07_as_10,restricaoSABADO_10_as_12#,
               #restricaoDOMINGO_07_as_10,restricaoDOMINGO_10_as_12):
               ):

    
        self.PROFESSORES = professores
        self.DISCIPLINAS = disciplinas
        self.DOMINIO = dominio
        self.RESTRICOES_AULA_DUPLA = restricoes_aula_dupla
        self.RESTRICOES_AULAS = restricoes_aulas
        self.escolha = escolha 
        self.OLD = {}
        self.contador=0
        self.comparacoes=0
        self.heuristica=escolha
        
        self.restricaoSEGUNDA_07_as_10 = restricaoSEGUNDA_07_as_10
        self.restricaoSEGUNDA_10_as_12 = restricaoSEGUNDA_10_as_12
        self.restricaoTERCA_07_as_10 = restricaoTERCA_07_as_10
        self.restricaoTERCA_10_as_12 = restricaoTERCA_10_as_12
        self.restricaoQUARTA_07_as_10 = restricaoQUARTA_07_as_10
        self.restricaoQUARTA_10_as_12 = restricaoQUARTA_10_as_12
        self.restricaoQUINTA_07_as_10 = restricaoQUINTA_07_as_10
        self.restricaoQUINTA_10_as_12 = restricaoQUINTA_10_as_12
        self.restricaoSEXTA_07_as_10 = restricaoSEXTA_07_as_10
        self.restricaoSEXTA_10_as_12 = restricaoSEXTA_10_as_12
        self.restricaoSABADO_07_as_10 = restricaoSABADO_07_as_10
        self.restricaoSABADO_10_as_12 = restricaoSABADO_10_as_12
        #self.restricaoDOMINGO_07_as_10 = restricaoDOMINGO_07_as_10
        #self.restricaoDOMINGO_10_as_12 = restricaoDOMINGO_10_as_12
    
    
#-----------------------------------------------------------
  
#-----------------------------------------------------------

#######################################
### Funcao para BACKTRACKING de CSP ###
#######################################
def BACKTRACKING_SEARCH(csp):
        
    #copia de dominio para OLD
    #csp.OLD = copy.deepcopy(csp.DOMINIO) 
    
    #inicializa resultado
    resultado={
    SEGUNDA_07_as_10:["--x--"],SEGUNDA_10_as_12:["--x--"],
    TERCA_07_as_10:["--x--"],TERCA_10_as_12:["--x--"],
    QUARTA_07_as_10:["--x--"],QUARTA_10_as_12:["--x--"],
    QUINTA_07_as_10:["--x--"],QUINTA_10_as_12:["--x--"],
    SEXTA_07_as_10:["--x--"],SEXTA_10_as_12:["--x--"],
    SABADO_07_as_10:["--x--"],SABADO_10_as_12:["--x--"]#,
    #DOMINGO_07_as_10:["--x--"],DOMINGO_10_as_12:["--x--"]
    }
   
    resultado= RECURSIVE_BACKTRACKING(resultado,csp)
    print " "
    print " busca: " + str(csp.contador)
    print " comparacoes: " + str(csp.comparacoes)    
    print " "
    
    return resultado

#-----------------------------------------------------------
  
#-----------------------------------------------------------
 
#####
#CSP#
#####
def RECURSIVE_BACKTRACKING(resultado,csp):
    
    #se todas as disciplinas foram alocadas, chega-se ao final
    if COMPLETO(resultado,csp):return resultado
    csp.contador +=1
    
    #para cada valor  das disciplinas (ja checado... so aparecem disciplinas validas... aquelas q ainda faltam alocacao) 
    for valor in DisciplinasForwardChecking(resultado,csp):
        print valor
        
        csp.comparacoes += 1

        #aqui eu coloco o periodo alocado pra fora de acordo com a heuristica e passo pra baixo
        if csp.heuristica == 1 or csp.heuristica == 2 or csp.heuristica == 3:
            periodo_alocacao= PegaPeriodoVazio(resultado)
        elif csp.heuristica == 4 or csp.heuristica == 6  :
            periodo_alocacao= PegaPeriodoRestringido(resultado,csp)
        elif csp.heuristica == 5 or csp.heuristica == 7  :
            periodo_alocacao= PegaPeriodoMenosRestringido(resultado, csp)
            
        #escolhido um valor
        #verifica se ele ficaria consistente com o restante dos assinalados
        if CONSISTENTE(valor,resultado,csp,periodo_alocacao):
#            
            #CONSISTENTE -> joga o valor la dentro 
            #periodo_alocacao= PegaPeriodoVazio(resultado)
            AlocaDisciplina(resultado, valor, periodo_alocacao)
            
            ####
            #if (periodo_alocacao[:10] == "SEGUNDA_07"):
            #    print "ss"
            #if (periodo_alocacao[:10] == "SEGUNDA_10"):
            #    print "xx"
            ####
             
            # ForwardChecking(valor, csp.DOMINIO)
             
            result=RECURSIVE_BACKTRACKING(resultado,csp)
            
            #volta da recursao
            #se estiver completo, acaba!
            if COMPLETO(result,csp):return result
            
            #se nao estiver completo.... loop
            else: 
                DesAlocaDisciplina(resultado, valor, periodo_alocacao)
            #     csp.DOMINIO = copy.deepcopy(csp.OLD)
             
        else:
            print '<- X (nao consistente)'   

    return resultado
 
 

#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
############################################### 
#verifica a consistencia do valor no resultado#
###############################################   
def CONSISTENTE(valor,resultado,csp,periodo_alocacao):

    for k in csp.DOMINIO[periodo_alocacao]: #aqui eu tenho a disciplina na mao ... dentre varias daquele item de dominio(Segunda_07_10, por exemplo)
        
        #verifica se o valor esta dentro do dominio de valores daquele periodo
        if valor == k:
        
            #verifica primeira restricao (AULA DUPLA)
            if csp.RESTRICOES_AULA_DUPLA[periodo_alocacao](periodo_alocacao,valor,resultado):
                return False
            else:
                #return True                        
                #aqui vou verificar a segunda restricao
                if csp.RESTRICOES_AULAS[periodo_alocacao](periodo_alocacao,valor, csp):
                    return False
                else: return True
  
#    #rodo por todo o dominio
#    for j,i in csp.DOMINIO.iteritems():
#        
#        #aqui eu tenho o slot do periodo na mao...
#        #if j ja esta preenchido no resultado :- sai fora :
#        if (resultado[j][0] == "--x--"):
#            
#            for k in i: #aqui eu tenho a disciplina na mao ... dentre varias daquele item de dominio(Segunda_07_10, por exemplo)
#                
#                #verifica se o valor esta dentro do dominio de valores daquele periodo
#                if valor == k:
#                
#                    #verifica primeira restricao (AULA DUPLA)
#                    if csp.RESTRICOES_AULA_DUPLA[j](j,valor,resultado):
#                        return False
#                    else:
#                        #return True                        
#                        #aqui vou verificar a segunda restricao
#                        if csp.RESTRICOES_AULAS[j](j,valor, csp):
#                            return False
#                        else: return True
# 
    return True


#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
############################## 
#pego s slot mais restringido#
##############################   
def PegaPeriodoRestringido(resultado,csp):

        ##heuristica MAIS RESTRINGIDA:pega o slot onde tem mais gente q nao pode e comeco com ele 
        
        slot=[]
        #rodo por todo o dominio
        for j,i in csp.DOMINIO.iteritems():
            
            #aqui eu tenho o slot do periodo VAZIO na mao...
            #if j ja esta preenchido no resultado :- sai fora :
            if (resultado[j][0] == "--x--"):
                
                n=NumRestricoesPeriodo(j, csp)
                #monto a tupla
                aux=(j,n)
                slot.append(aux)
        
        #ordeno a lista
        sorteado=sorted(slot, key=lambda tup: tup[1], reverse=True) 
        
        return sorteado[0][0]


#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
############################### 
#pego s slot menos restringido#
###############################   
def PegaPeriodoMenosRestringido(resultado,csp):

        ##heuristica MAIS RESTRINGIDA:pega o slot onde tem mais gente q nao pode e comeco com ele 
        
        slot=[]
        #rodo por todo o dominio
        for j,i in csp.DOMINIO.iteritems():
            
            #aqui eu tenho o slot do periodo VAZIO na mao...
            #if j ja esta preenchido no resultado :- sai fora :
            if (resultado[j][0] == "--x--"):
                
                n=NumRestricoesPeriodo(j, csp)
                #monto a tupla
                aux=(j,n)
                slot.append(aux)
        
        #ordeno a lista
        sorteado=sorted(slot, key=lambda tup: tup[1]) 
        
        return sorteado[0][0]



#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
######################################## 
#realiza checagem se resultado terminou#
########################################   
def COMPLETO(resultado,csp):
    
    #conta o numero de disciplinas que precisam ser alocadas.
    num_disciplinas=0
    for i in csp.DISCIPLINAS:
        num_disciplinas += 1
    #multiplica por X, pois cada aula sera dada X vezes na semana
    num_disciplinas = num_disciplinas * qtde_aulas_semana
    
    #conta o numero de resultados encontrados
    num_resultados=0
    for i in resultado:
        for j in resultado[i][:]:
            if j!="--x--": num_resultados +=1   
     
    if(num_resultados>=num_disciplinas): return True
        
    return False
    
#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
 
######################################## 
#realiza checagem se resultado terminou#
########################################   
def COMPLETOFINAL(resultado):
    
    #conta o numero de disciplinas que precisam ser alocadas.
    num_disciplinas=qtde_disciplinas*2
    
    #conta o numero de resultados encontrados
    num_resultados=0
    for i in resultado:
        for j in resultado[i][:]:
            if j!="--x--": num_resultados +=1   
     
    if(num_resultados>=num_disciplinas): return True
        
    return False
    


#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
############################### 
#traz as possiveis disciplinas#
###############################   
def DisciplinasForwardChecking(resultado,csp):
    #da variavel CSP eu puxo as disciplinas
    #da variavel resultado eu puxo o periodo em q estou olhando... pra isso eu consigo  ver preferencias ou heuristicas...
    
    disciplinas = []
    retorno=[]
    
    #para cada disciplina
    for i in csp.DISCIPLINAS:
        #checagem se a disciplina ja foi alocada X vezes no resultado
        if not TerminoDisciplina(i,resultado):
            disciplinas.append(i)  

    if csp.heuristica == 1 or csp.heuristica == 4 or csp.heuristica == 5:    
        
        #escolho uma ordem randomica das disciplinas q sobraram
        for j in range(len(disciplinas)):
            elemento = random.choice(disciplinas)
            disciplinas.remove(elemento)
            retorno.append(elemento)
    
    elif csp.heuristica == 2 or csp.heuristica == 6:
        #heuristica MAIS RESTRITIVA: acho q esse nao da... forcando um pouquinho. da..
        #pega a disciplina cujo professor tem mais restricao e comeca por ela
        # essa fica maneira: pego a disc. do(s) professor q tem mais restricao no calendario

        #aqui tenho as disciplinas na mao...
        disciplinaH = []
        professor=""
        for i in disciplinas:
            #descobre o professor daquela disciplina
            for j,k in csp.PROFESSORES.iteritems():
                for x in k:
                    if x == i:
                        professor=j
                        break
                    
            #recupero o numero de vezes que esse professor aparece nas restricoes
            n=NumRestricoesProfessor(professor, csp)
            
            #monto a tupla
            aux=(i,n)
            disciplinaH.append(aux)
        #ordeno a lista
        sorteado=sorted(disciplinaH, key=lambda tup: tup[1], reverse=True) 

        #recupero a lista ordenada pra dentro da resposta
        for i in sorteado:
            retorno.append(i[0]) 

    
    elif csp.heuristica == 3 or csp.heuristica == 7:
        ##heuristica MENOS RESTRITIVA: acho q esse nao da... forcando um pouquinho. da..
        #pega a disciplina cujo professor tem mmenos restricao e comeca por ela
        ## essa fica maneira: pego a disc. do(s) professor q tem menos restricao no calendario

        #aqui tenho as disciplinas na mao...
        disciplinaH = []
        professor=""
        for i in disciplinas:
            #descobre o professor daquela disciplina
            for j,k in csp.PROFESSORES.iteritems():
                for x in k:
                    if x == i:
                        professor=j
                        break
                    
            #recupero o numero de vezes que esse professor aparece nas restricoes
            n=NumRestricoesProfessor(professor, csp)
            
            #monto a tupla
            aux=(i,n)
            disciplinaH.append(aux)
        #ordeno a lista
        sorteado=sorted(disciplinaH, key=lambda tup: tup[1]) 

        #recupero a lista ordenada pra dentro da resposta
        for i in sorteado:
            retorno.append(i[0]) 

        
    return retorno



#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
################################################ 
#Conta o numero de Restricoes q o professor tem#
################################################   
def NumRestricoesPeriodo(periodo, csp):

    n=0
    
    if periodo[:3] == "SEG" and periodo[-2:] == "10" :
        for i in   csp.restricaoSEGUNDA_07_as_10: 
            n +=1
    elif periodo[:3] == "SEG" and periodo[-2:] == "12" :
        for i in   csp.restricaoSEGUNDA_10_as_12: 
            n +=1

    elif periodo[:3] == "TER" and periodo[-2:] == "10" :
        for i in   csp.restricaoTERCA_07_as_10: 
            n +=1
    elif periodo[:3] == "TER" and periodo[-2:] == "12" :
        for i in   csp.restricaoTERCA_10_as_12: 
            n +=1

    elif periodo[:3] == "QUA" and periodo[-2:] == "10" :
        for i in   csp.restricaoQUARTA_07_as_10: 
            n +=1
    elif periodo[:3] == "QUA" and periodo[-2:] == "12" :
        for i in   csp.restricaoQUARTA_10_as_12: 
            n +=1

    elif periodo[:3] == "QUI" and periodo[-2:] == "10" :
        for i in   csp.restricaoQUINTA_07_as_10: 
            n +=1
    elif periodo[:3] == "QUI" and periodo[-2:] == "12" :
        for i in   csp.restricaoQUINTA_10_as_12: 
            n +=1


    elif periodo[:3] == "SEX" and periodo[-2:] == "10" :
        for i in   csp.restricaoSEXTA_07_as_10: 
            n +=1
    elif periodo[:3] == "SEX" and periodo[-2:] == "12" :
        for i in   csp.restricaoSEXTA_10_as_12: 
            n +=1


    elif periodo[:3] == "SAB" and periodo[-2:] == "10" :
        for i in   csp.restricaoSABADO_07_as_10: 
            n +=1
    elif periodo[:3] == "SAB" and periodo[-2:] == "12" :
        for i in   csp.restricaoSABADO_10_as_12: 
            n +=1

  
            
    return n

#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
################################################ 
#Conta o numero de Restricoes q o professor tem#
################################################   
def NumRestricoesProfessor(professor, csp):
    n=0
    
    if professor in csp.restricaoSEGUNDA_07_as_10: n +=1
    if professor in csp.restricaoSEGUNDA_10_as_12: n +=1
    if professor in csp.restricaoTERCA_07_as_10: n +=1
    if professor in csp.restricaoTERCA_10_as_12: n +=1
    if professor in csp.restricaoQUARTA_07_as_10: n +=1
    if professor in csp.restricaoQUARTA_10_as_12: n +=1
    if professor in csp.restricaoQUINTA_07_as_10: n +=1
    if professor in csp.restricaoQUINTA_10_as_12: n +=1
    if professor in csp.restricaoSEXTA_07_as_10: n +=1
    if professor in csp.restricaoSEXTA_10_as_12: n +=1
    if professor in csp.restricaoSABADO_07_as_10: n +=1
    if professor in csp.restricaoSABADO_10_as_12: n +=1
        
    
    return n

#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
##################################### 
#Aloca a disciplina no periodo vazio#
#####################################   
def AlocaDisciplina(resultado, valor, periodo_alocacao):
    
        
        #rodo por todo o dominio
#        for j,i in resultado.iteritems():
#            
#            #aqui eu tenho o slot do periodo na mao...
#            #if j ja esta preenchido no resultado :- sai fora :
#            if (resultado[j][0] == "--x--"):
#                
#                #aqui tenho o primeiro slot vazio.
#                #prego o valor
#                resultado[j][0] =valor
#                return    
    
    resultado[periodo_alocacao][0]=valor  
    return



#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
##################################### 
#Aloca a disciplina no periodo vazio#
#####################################   
def DesAlocaDisciplina(resultado, valor, periodo_alocacao):
    
        
        #rodo por todo o dominio
#        for j,i in resultado.iteritems():
#            
#            #aqui eu tenho o slot do periodo na mao...
#            #if j ja esta preenchido no resultado :- sai fora :
#            if (resultado[j][0] == "--x--"):
#                
#                #aqui tenho o primeiro slot vazio.
#                #prego o valor
#                resultado[j][0] =valor
#                return    
    
    resultado[periodo_alocacao][0]="--x--"  
    return

#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
##################################### 
#Aloca a disciplina no periodo vazio#
#####################################   
def PegaPeriodoVazio(resultado):
        
        #rodo por todo o dominio
        for j,i in resultado.iteritems():
            
            #aqui eu tenho o slot do periodo na mao...
            #if j ja esta preenchido no resultado :- sai fora :
            if (resultado[j][0] == "--x--"):
                
                #aqui tenho o primeiro slot vazio.
                return j    
    
        return

#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
###################################################################### 
#realiza checagem se a disciplina ja foi alocada no resultado X vezes#
######################################################################   
def TerminoDisciplina(disciplina,resultado):

    num_disciplina=0
    #percorre o resultado e verifica quantas vezes aparece a disciplina em questao
    for i in resultado:
        for j in resultado[i][:]:
            if j == disciplina:
                num_disciplina += 1  
    
    #se a disciplina aparecer o mesmo numero de vezes que a constante QTDE_AULAS_SEMANA :- entao ela ja terminou... nao 
    #eh mais candidata para a solucao
    if num_disciplina >=qtde_aulas_semana: return True
     
    return False

#-----------------------------------------------------------
  
#-----------------------------------------------------------

 
################################## 
#realiza checagem de aulas duplas#
##################################   
def restricaoAULADUPLA(periodo, valor,resultado):

    periodo_aux="--x--"
    #periodo2=""
    
    #pega a substring do periodo a ser inserido
    substring=periodo[:3]
    
    #procura no resultado pelo mesmo periodo 1 e 2 no mesmo dia
    for i in resultado:
        if i[:3] == substring:
            #estou no mesmo bloco do periodo (periodo 1 e 2)
            if i != periodo:
                periodo_aux=resultado[i][0]
            break #posso pular fora sem fazer o loop

    #tenho na mao o valor e o periodo anterior ou proximo (dependendo do caso)
    #nao importa, eu posso comparar.
    if periodo_aux == valor :
        return True
            
    return False

#-----------------------------------------------------------
  
#-----------------------------------------------------------

######################################## 
#realiza checagem de aulas na restricao#
########################################   
def restricaoAULA(j,valor, csp):
    
    #switch para pegar a variavel correta dentro do CSP
    restricao=[]
    if    j == SEGUNDA_07_as_10: restricao = csp.restricaoSEGUNDA_07_as_10
    elif  j == SEGUNDA_10_as_12: restricao = csp.restricaoSEGUNDA_10_as_12

    elif  j == TERCA_07_as_10: restricao = csp.restricaoTERCA_07_as_10
    elif  j == TERCA_10_as_12: restricao = csp.restricaoTERCA_10_as_12

    elif  j == QUARTA_07_as_10: restricao = csp.restricaoQUARTA_07_as_10
    elif  j == QUARTA_10_as_12: restricao = csp.restricaoQUARTA_10_as_12

    elif  j == QUINTA_07_as_10: restricao = csp.restricaoQUINTA_07_as_10
    elif  j == QUINTA_10_as_12: restricao = csp.restricaoQUINTA_10_as_12

    elif  j == SEXTA_07_as_10: restricao = csp.restricaoSEXTA_07_as_10
    elif  j == SEXTA_10_as_12: restricao = csp.restricaoSEXTA_10_as_12

    elif  j == SABADO_07_as_10: restricao = csp.restricaoSABADO_07_as_10
    elif  j == SABADO_10_as_12: restricao = csp.restricaoSABADO_10_as_12

    #elif  j == DOMINGO_07_as_10: restricao = csp.restricaoDOMINGO_07_as_10
    #elif  j == DOMINGO_10_as_12: restricao = csp.restricaoDOMINGO_10_as_12

    #com a lista dos professores que possuem restricao para esse periodo na mao... eu pego e vejo se aquela disciplina eh dada por algum deles
    for i in restricao:
        #para cada restricao de materia do professor (i)
        for j in csp.PROFESSORES[i]:
            if j == valor:
                return True
     
     
    return False

#-----------------------------------------------------------
  
#-----------------------------------------------------------

##############################################
### imprime o resultado na ordem da semana ###
##############################################
def Imprime(resultado):


    #inacreditavel ter q fazer isso !!!!!!    
    #nao da tempo de fazer uma funcao de sort q traga na ordem da semana
    for i,j in resultado.iteritems():
        if i == "SEGUNDA_07_as_10": 
            print "SEGUNDA_07_as_10: " + resultado[i][0]
            break
    for i,j in resultado.iteritems():
        if i == "SEGUNDA_10_as_12": 
            print "SEGUNDA_10_as_12: " + resultado[i][0]
            break

    for i,j in resultado.iteritems():
        if i == "TERCA_07_as_10": 
            print "TERCA_07_as_10: " + resultado[i][0]
            break
    for i,j in resultado.iteritems():
        if i == "TERCA_10_as_12": 
            print "TERCA_10_as_12: " + resultado[i][0]
            break

    for i,j in resultado.iteritems():
        if i == "QUARTA_07_as_10": 
            print "QUARTA_07_as_10: " + resultado[i][0]
            break
    for i,j in resultado.iteritems():
        if i == "QUARTA_10_as_12": 
            print "QUARTA_10_as_12: " + resultado[i][0]
            break

    for i,j in resultado.iteritems():
        if i == "QUINTA_07_as_10": 
            print "QUINTA_07_as_10: " + resultado[i][0]
            break
    for i,j in resultado.iteritems():
        if i == "QUINTA_10_as_12": 
            print "QUINTA_10_as_12: " + resultado[i][0]
            break

    for i,j in resultado.iteritems():
        if i == "SEXTA_07_as_10": 
            print "SEXTA_07_as_10: " + resultado[i][0]
            break
    for i,j in resultado.iteritems():
        if i == "SEXTA_10_as_12": 
            print "SEXTA_10_as_12: " + resultado[i][0]
            break
    for i,j in resultado.iteritems():
        if i == "SABADO_07_as_10": 
            print "SABADO_07_as_10: " + resultado[i][0]
            break
    for i,j in resultado.iteritems():
        if i == "SABADO_10_as_12": 
            print "SABADO_10_as_12: " + resultado[i][0]
            break
    #for i,j in resultado.iteritems():
    #    if i == "DOMINGO_07_as_10": 
    #        print "DOMINGO_07_as_10: " + resultado[i][0]
    #        break
    #for i,j in resultado.iteritems():
    #    if i == "DOMINGO_10_as_12": 
    #        print "DOMINGO_10_as_12: " + resultado[i][0]
    #        break
        
#-----------------------------------------------------------
  
#-----------------------------------------------------------


######################
### Classe de Aula ###
######################
#Retorna uma configuracao de aulas
def ClasseAula(escolha):
    
    ####################################
    #disciplinas (nome das disciplinas)#
    ####################################
    disciplinas=[inteligencia_artificial, banco_de_dados, calculo, sma, algoritmo,engenharia_software, metodologia_cientifica]
    disciplinas=[inteligencia_artificial, banco_de_dados, calculo, algoritmo, metodologia_cientifica]
    disciplinas=[inteligencia_artificial, banco_de_dados, calculo, algoritmo, metodologia_cientifica, engenharia_software]

    #######################################################
    #dominios = todos os valores que a variavel pode tomar#
    #######################################################
    #dominio={SEGUNDA:[disciplinas][disciplinas], TERCA:[disciplinas][disciplinas], QUARTA:[disciplinas][disciplinas], QUINTA:[disciplinas][disciplinas], SEXTA:[disciplinas][disciplinas], SABADO:[disciplinas][disciplinas], DOMINGO:[disciplinas][disciplinas]}
    dominio={
            SEGUNDA_07_as_10:disciplinas[:],SEGUNDA_10_as_12:disciplinas[:],
            TERCA_07_as_10:disciplinas[:],TERCA_10_as_12:disciplinas[:],
            QUARTA_07_as_10:disciplinas[:],QUARTA_10_as_12:disciplinas[:],
            QUINTA_07_as_10:disciplinas[:],QUINTA_10_as_12:disciplinas[:],
            SEXTA_07_as_10:disciplinas[:],SEXTA_10_as_12:disciplinas[:],
            SABADO_07_as_10:disciplinas[:],SABADO_10_as_12:disciplinas[:]#,
            #DOMINGO_07_as_10:disciplinas[:],DOMINGO_10_as_12:disciplinas[:]
            }
        
    #####################################
    #aqui coloco a restricao de cada dia#
    #####################################
    restricoes_aula_dupla={
                      SEGUNDA_07_as_10:restricaoAULADUPLA,SEGUNDA_10_as_12:restricaoAULADUPLA,
                      TERCA_07_as_10:restricaoAULADUPLA,TERCA_10_as_12:restricaoAULADUPLA,
                      QUARTA_07_as_10:restricaoAULADUPLA,QUARTA_10_as_12:restricaoAULADUPLA,
                      QUINTA_07_as_10:restricaoAULADUPLA,QUINTA_10_as_12:restricaoAULADUPLA,
                      SEXTA_07_as_10:restricaoAULADUPLA,SEXTA_10_as_12:restricaoAULADUPLA,
                      SABADO_07_as_10:restricaoAULADUPLA,SABADO_10_as_12:restricaoAULADUPLA#,
                      #DOMINGO_07_as_10:restricaoAULADUPLA,DOMINGO_10_as_12:restricaoAULADUPLA
                     }    

    restricoes_aulas={
                      SEGUNDA_07_as_10:restricaoAULA,SEGUNDA_10_as_12:restricaoAULA,
                      TERCA_07_as_10:restricaoAULA,TERCA_10_as_12:restricaoAULA,
                      QUARTA_07_as_10:restricaoAULA,QUARTA_10_as_12:restricaoAULA,
                      QUINTA_07_as_10:restricaoAULA,QUINTA_10_as_12:restricaoAULA,
                      SEXTA_07_as_10:restricaoAULA,SEXTA_10_as_12:restricaoAULA,
                      SABADO_07_as_10:restricaoAULA,SABADO_10_as_12:restricaoAULA#,
                      #DOMINGO_07_as_10:restricaoAULA,DOMINGO_10_as_12:restricaoAULA
                     }    
    
#    restricoes_aulas={
#                      SEGUNDA_07_as_10:restricaoSEGUNDA_07_10,SEGUNDA_10_as_12:restricaoSEGUNDA_10_12,
#                      TERCA_07_as_10:restricaoTERCA_07_10,TERCA_10_as_12:restricaoTERCA_10_12,
#                      QUARTA_07_as_10:restricaoQUARTA_07_10,QUARTA_10_as_12:restricaoQUARTA_10_12,
#                      QUINTA_07_as_10:restricaoQUINTA_07_10,QUINTA_10_as_12:restricaoQUINTA_10_12,
#                      SEXTA_07_as_10:restricaoSEXTA_07_10,SEXTA_10_as_12:restricaoSEXTA_10_12,
#                      SABADO_07_as_10:restricaoSABADO_07_10,SABADO_10_as_12:restricaoSABADO_10_12,
#                      DOMINGO_07_as_10:restricaoDOMINGO_07_10,DOMINGO_10_as_12:restricaoDOMINGO_10_12
#                     }


















    #-------------------------------------------------
    #                SECAO PARA CONFIGURACAO
    #-------------------------------------------------

    #IMPORTANTE: assumi que cada disciplina so tera 2 aulas por semana
    #IMPORTANTE: 1 professor por disciplina.
    
    #########################
    #disciplinas professores#
    #########################
    #essa variavel e por causa da restricao de dupla aula no mesmo dia... caso nao tivesse nao seria necessaria
    professores={Ana_Cristina:[inteligencia_artificial, metodologia_cientifica], Ferraz:[algoritmo,calculo], Plastino:[banco_de_dados], Viviane:[engenharia_software,sma] }
    professores={Ana_Cristina:[inteligencia_artificial, metodologia_cientifica], Ferraz:[algoritmo,calculo], Plastino:[banco_de_dados] }
    professores={Ana_Cristina:[inteligencia_artificial, metodologia_cientifica], Ferraz:[algoritmo,calculo], Plastino:[banco_de_dados], Viviane:[engenharia_software] }
    
    ########################
    #Restricoes por periodo#
    ########################
    restricaoSEGUNDA_07_as_10=[Plastino]
    restricaoSEGUNDA_10_as_12=[Ana_Cristina]
    
    restricaoTERCA_07_as_10=[Ferraz]
    restricaoTERCA_10_as_12=[Viviane,Ferraz]
    
    restricaoQUARTA_07_as_10=[]
    restricaoQUARTA_10_as_12=[Plastino]
    
    restricaoQUINTA_07_as_10=[Ferraz]
    restricaoQUINTA_10_as_12=[Viviane]
    
    restricaoSEXTA_07_as_10=[Ana_Cristina,Ferraz]
    restricaoSEXTA_10_as_12=[Ana_Cristina,Ferraz]
    
    restricaoSABADO_07_as_10=[Ana_Cristina,Ferraz]
    restricaoSABADO_10_as_12=[Ana_Cristina,Ferraz]
    
    #-------------------------------------------------
    #-------------------------------------------------

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    return csp( professores,disciplinas,dominio,restricoes_aula_dupla,restricoes_aulas,escolha,
               restricaoSEGUNDA_07_as_10,restricaoSEGUNDA_10_as_12,
               restricaoTERCA_07_as_10,restricaoTERCA_10_as_12,
               restricaoQUARTA_07_as_10,restricaoQUARTA_10_as_12,
               restricaoQUINTA_07_as_10,restricaoQUINTA_10_as_12,
               restricaoSEXTA_07_as_10,restricaoSEXTA_10_as_12,
               restricaoSABADO_07_as_10,restricaoSABADO_10_as_12#,
               #restricaoDOMINGO_07_as_10,restricaoDOMINGO_10_as_12
               )

##############################################################################################
##############################################################################################
##############################################################################################



########################
## roda SOLVER do CSP ##
########################
def run():
 
    #pede para escolher o algoritmo de CSP com ou sem heuristica 
    escolha="0"
    while not (int(escolha) > 0 and int(escolha) < 8):
        print " "
        print "(1) para CSP sem Heuristica"
        print "(2) para CSP com heuristica MAIS RESTRITIVA"
        print "(3) para CSP com heuristica MENOS RESTRITIVA "
        print "(4) para CSP com heuristica MAIS RESTRINGIDA"
        print "(5) para CSP com heuristica MENOS RESTRINGIDA"
        print "(6) para CSP com heuristica MAIS RESTRINGIDA com MAIS RESTRITIVA"
        print "(7) para CSP com heuristica MENOS RESTRINGIDA com MENOS RESTRITIVA"
        
        escolha = raw_input("escolha o algoritmo: ")

    resultado =  BACKTRACKING_SEARCH(ClasseAula(int(escolha)))  
    
    print "###############"
    print "## resultado ##"
    print "###############"
    if COMPLETOFINAL(resultado) :
        Imprime(resultado)
    else:
        print " "
        print "NAO eh possivel resolver o problema"
    print " "
        


#Inicia o Programa
run()
