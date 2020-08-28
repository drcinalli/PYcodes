'''
Created on 21/04/2012

@author: quatrosem
'''

import random
import copy
   
Preco = {'SOPA_CANJA':10,'SOPA_FEIJAO':10,'SOPA_HOT_SOUR':14,'SOPA_MINESTRONE':15,'SOPA_LEGUMES':10, 'SOPA_PEDRA':13,'FRANGO_XADREZ':30,'GALINHA_MOLHO_PARDO':45,'FRANGO_AMENDOCREAM':22, 'FRANGO_FRITO':24, 'FRANGO_ASSADO':18,'SALADA_CEASAR':15,'SALADA_TROPICAL':19,'SALADA_MIX':22,'QUEIJO_SALAME':9,'ARROZ_BRANCO':12,'CAMARAO_A_GREGA':45,'OSTRAS':55,'LULA_E_MARISCO':45,'PICANHA_SUINA':36,'COSTELINHA':29, 'PE_DE_PORCO':15, 'LOMBINHO':29,'CARRE':25,  'BACON':22, 'TORRESMO':15, 'PEIXE_ASSADO':35, 'CONGRIO_ROSA':36, 'POLVO':32, 'PAELLA':50, 'PIRAO':23, 'ARROZ_MALUCO':12, 'ARROZ_A_GREGA':18, 'ARROZ_LENTILHA':16, 'ARROZ_FAROFA':14, 'ARROZ_FEIJAO':14, 'AZEITONA_QUEIJO':12, 'TORRADA_COM_PATE':8, 'COUVERT_COMPLETO':19, 'ISCA_DE_CARNE':26, 'PASTEIS':19, 'PAO_PATE_MANTEIGA':17, 'ESPETINHO':26, 'SALADA_TOMATE':15, 'SALADA_VINAGRETTE':16, 'SALADA_PALMITO':25, 'SALADA_ALLHO':12, 'FRANGO_MADEIRA':34, 'FRANGO_MAIONESE':28}
CONTAMAX=350


#################
#valores
# nao era global, mas coloquei pq descobri no final que estava misturando porco com fruto do mar....
#isso  porque tive de mexer antes no dicionario e escangalhei as chaves. antes elas eram tipo de prato...
#desde q coloquei o preco ... mude a configuracao toda delas.... agora a chave e um hash diferente.
# ai nao funcionava a minha restricao do porco e do Fruto do mar.
#veja la o comentario de como era antes .... era bonito.
#
# agora nao quero mexer em tudo de novo pra passar o CSP pra restricao e tal ... muito trabalho, melhor ler a global aqui e prnto.
# peco desculpas pelo talho, mas o objetivo maior eh a tecnica e o algoritmo ... nao a programacao
valoresSOPA=['SOPA_CANJA','SOPA_FEIJAO','SOPA_HOT_SOUR', 'SOPA_MINESTRONE','SOPA_LEGUMES', 'SOPA_PEDRA']
valoresGALINHA=['FRANGO_XADREZ','GALINHA_MOLHO_PARDO','FRANGO_AMENDOCREAM', 'FRANGO_FRITO', 'FRANGO_ASSADO', 'FRANGO_MADEIRA', 'FRANGO_MAIONESE']
valoresVEGETAIS=['SALADA_CEASAR','SALADA_TROPICAL','SALADA_MIX', 'SALADA_TOMATE', 'SALADA_VINAGRETTE', 'SALADA_PALMITO', 'SALADA_ALLHO']
valoresAPERITIVO=['QUEIJO_SALAME', 'AZEITONA_QUEIJO', 'TORRADA_COM_PATE', 'COUVERT_COMPLETO', 'ISCA_DE_CARNE', 'PASTEIS', 'PAO_PATE_MANTEIGA', 'ESPETINHO']
#valoresAPERITIVO=[]
#valoresVEGETAIS=[]
#valoresGALINHA=[]
valoresARROZ=['ARROZ_BRANCO', 'ARROZ_MALUCO', 'ARROZ_A_GREGA', 'ARROZ_LENTILHA', 'ARROZ_FAROFA', 'ARROZ_FEIJAO']
valoresFRUTOSMAR=['CAMARAO_A_GREGA','OSTRAS','LULA_E_MARISCO', 'PEIXE_ASSADO', 'CONGRIO_ROSA', 'POLVO', 'PAELLA', 'PIRAO']
valoresPORCO=['PICANHA_SUINA','COSTELINHA', 'PE_DE_PORCO', 'LOMBINHO','CARRE', 'BACON', 'TORRESMO']




#classe para o CSP
class csp:
  def __init__(self, variaveis,dominio,restricoes,n):
    ## these remain constant during search
    self.VARIAVEIS = variaveis
    self.DOMINIO = dominio
    self.RESTRICOES = restricoes
    self.pratos = n #pq sei q havera uma sopa a mais
    self.OLD = {}
    
#-----------------------------------------------------------

#chama programa para calcular resultado CSP
def BACKTRACKING_SEARCH(csp):
    
    # aqui tenho q chamar a SOPA
    ###
    tipoprato="SOPA"
    resultado={}
    #para cada valor do dominio daquela variavel (prato)
    #
    # Essa parte ficou feia ... esta forcado demais a entrega da sopa...
    #... preciso pensar e mudar isso depois
    #
    for valor in PratosSOPA(tipoprato,resultado,csp):
         #print valor
         #escolhido um valor
         #verifica se ele ficaria consistente com o restante dos assinalados
         if CONSISTENTE(valor,resultado,csp.RESTRICOES, csp.DOMINIO): # aqui ele aceita sempre o primeiro, mas nao tem problema porque sera reescrito pelo hotsour de qquer maneira... e se ele for o primeiro, nenhum outro vai substituir
#            #se pode esse prato 
             resultado[tipoprato]=valor
             #procurar comando break para sair do loop
    
    #copia de dominio para OLD
    csp.OLD = copy.deepcopy(csp.DOMINIO) 
    
    return RECURSIVE_BACKTRACKING(resultado,csp,1)


#CSP
def RECURSIVE_BACKTRACKING(resultado,csp,cont):
    
    #se o total de pratos foi atingido
    if COMPLETO(resultado,csp):return resultado
    
    #buca prox variavel
    #tipoprato=ProxVariavel( csp.VARIAVEIS,resultado,csp)
    
    
#    #para cada valor do dominio daquela variavel (prato)
    #for valor in PratosVariavel(tipoprato,resultado,csp):
    for valor in PratosVariavel(resultado,csp):
         print valor
         #escolhido um valor
         #verifica se ele ficaria consistente com o restante dos assinalados
         if CONSISTENTE(valor,resultado,csp.RESTRICOES, csp.DOMINIO):
#            #se pode esse prato 
             resultado[valor+str(cont)]=valor
             #resultado[[k for k, v in csp.DOMINIO.iteritems() if v == val][0]+str(cont)]=valor
             
             #aquiiiiiiiiiii AC3
             #csp.DOMINIO[valor+str(cont)]=valor
             
             ForwardChecking(valor, csp.DOMINIO)
             
             result=RECURSIVE_BACKTRACKING(resultado,csp,cont+1)
             #volta da recursao
             if COMPLETO(result,csp):return result
             else: 
                 del resultado[valor+str(cont)]
                 csp.DOMINIO = copy.deepcopy(csp.OLD)
             
             #if result!='falha':return result
             #del resultado[tipoprato]
         else:
             print '<- X'
    
    #se nao fica consistente, ele volta mudando os pratos anteriores
    #esse algoritmo nao faz o forward checking....
    #return "falha"
    return resultado

#pega a proxima variavel da lista
def ProxVariavel(vars,resultado,csp):
    
    random.shuffle(vars)
    
    for var in vars:
        if not (var in resultado):return var


#verifica se a lista de resultados esta completa  
def COMPLETO(resultado,csp):
    i=0
    for var in resultado:
        i += 1
    
    if i == csp.pratos: return True   
    return False

def select(data):
    if data != []:
        index = random.randint(0, len(data) - 1)
        elem = data[index]
        data[index] = data[-1]
        del data[-1]
        return elem
    else:
        return data

# Traz os dominios daquela variavel
def PratosVariavel(resultado,csp):
    #return csp.DOMINIO[var][:]
    volta = []
    result=[]
#    for i in csp.DOMINIO[var][:]:
#        volta.append(i)  
    for i in csp.DOMINIO:
        for k in csp.DOMINIO[i][:]:
            volta.append(k)  
    

    for j in range(len(volta)):
        element = random.choice(volta)
        volta.remove(element)
        result.append(element)
    
    return result

# Traz os dominios daquela variavel
def PratosSOPA(var, resultado,csp):
    #return csp.DOMINIO[var][:]
    volta = []
    result=[]
    for i in csp.DOMINIO[var][:]:
        volta.append(i)  

    for j in range(len(volta)):
        element = random.choice(volta)
        volta.remove(element)
        result.append(element)
 
 
            
    return result

#faz forward checking no dominio
def ForwardChecking(valor, dominio):

    for j,i in dominio.iteritems():
        for k in i:
            if valor == k:
                #del dominio[j]
                #aqui tenho o valor na mao(k) e tipo na outra (j).
                if k in valoresFRUTOSMAR: 
                    if 'PORCO' in dominio:
                        print '... forward PORCO'
                        del dominio['PORCO']
                        return
                elif k in valoresPORCO:
                    if 'FRUTOSMAR' in dominio:
                        print '... forward FRUTOSMAR'
                        del dominio['FRUTOSMAR']
                        return
                return
    #for j,i in mesa.iteritems():
    #    if i in valoresFRUTOSMAR: return False     

# Constraint Propagation with AC-3

def AC3(csp, queue=None):
    """[Fig. 5.7]"""
    if queue == None:
        queue = [(Xi, Xk) for Xi in csp.vars for Xk in csp.neighbors[Xi]]
    while queue:
        (Xi, Xj) = queue.pop()
        if remove_inconsistent_values(csp, Xi, Xj):
            for Xk in csp.neighbors[Xi]:
                queue.append((Xk, Xi))

def remove_inconsistent_values(csp, Xi, Xj):
    "Return true if we remove a value."
    removed = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        if every(lambda y: not csp.constraints(Xi, x, Xj, y),
                csp.curr_domains[Xj]):
            csp.curr_domains[Xi].remove(x)
            removed = True
    return removed

#verifica consistenca das respostas anteriores e dessa opcao
def CONSISTENTE(valor,resultado,restricoes, dominio):
    
    #se esta vazio, volta
    #if not resultado:return True
    
    #verifica prato repetido
    #for i in resultado:
    #    if resultado[i] == valor:
    #        return False
          ##nao vou fazer essa checagem agora nao
    
    #verifica se satisfaz a restricao
    #if not restricoes[tipoprato](valor,resultado):
    #    return False
    #[k for k, v in symbol_dic.iteritems() if v == val][0]
    #lKey = [key for key, value in dominio.iteritems() 
    #        if valor in value][0]
    for j,i in dominio.iteritems():
        for k in i:
            if valor == k:
                if not restricoes[j](valor,resultado):
                    return False
                else:
                    #print Conta(resultado) + Preco[valor]
                    if (Conta(resultado) + Preco[valor]) > CONTAMAX:
                        return False
                    else: return True
   
    #verifica se satisfaz a restricao de preco
#    print Conta(resultado) + Preco[valor]
#    if (Conta(resultado) + Preco[valor]) > CONTAMAX:
#        return False
        
    #nunca passara por aqui
    return False        

## ele nao precisaria checar todo o conjunto de constraints ... mas acho q fez isso didaticamente
## pq em tese vc deveria repassar todas as restricoes e ver se estao ok.
## nesse caso especdico nao tem sentido pq ele olha apenas o novo no e os vizinhos.


#Verifica se a sopa eh aquela permitida
def restricaoSOPA(sopa, mesa):
    
    if sopa == "SOPA_HOT_SOUR": return True
    
    return False

#Verifica se a galinha eh aquela permitida
def restricaoGALINHA(prato, mesa): 
    
    if (prato == "FRANGO_XADREZ") or (prato == "FRANGO_AMENDOCREAM"): return False
    
    return True

#Verifica se o vegetal eh aquele permitido
def restricaoVEGETAIS(prato, mesa): 
    
    if (prato == "SALADA_MIX"): return False
    
    return True


#Sem restircao
def restricaoAPERITIVO(prato, mesa):
    return True


#Sem restircao
def restricaoARROZ(prato, mesa):
    return True

#def FrutosMar(prato):
#    if (prato == "CAMARAO_A_GREGA") or (prato == "OSTRAS") or (prato == "LULA_E_MARISCO"): return True 
#    
#    return False
#
#def Porco(prato):
#    if (prato == "PICANHA_SUINA") or (prato == "COSTELINHA"): return True 
#    
#    return False

#Verifica restircao do Porco: se tiver porco nao pode ter frutos do Mar
def restricaoPORCO(prato, mesa): 
    
    
   # for i in mesa:
   #     if "FRUTOSMAR" == i: return False 
    for j,i in mesa.iteritems():
        if i in valoresFRUTOSMAR: return False     
    
    return True

#Verifica restircao do FrutosMar: se tiver porco nao pode ter frutos do Mar
def restricaoFRUTOSMAR(prato, mesa): 
    
   # for i in mesa:
   #    if "PORCO" == i: return False 
    for j,i in mesa.iteritems():
        if i in valoresPORCO: return False     
   
    return True

def Conta(resultado):
    total=0
    for i in resultado:
        total = total + Preco[resultado[i]]
    
    return total

###############
### Chines ###
###############
def Chines(n):#Retorna uma configuracao de pratos
    
    
    SOPA, GALINHA, VEGETAIS, APERITIVO, ARROZ, FRUTOSMAR, PORCO = "SOPA", "GALINHA", "VEGETAIS", "APERITIVO", "ARROZ", "FRUTOSMAR", "PORCO"
    
    
    
    #variaveis
    variaveis=[SOPA, GALINHA, VEGETAIS, APERITIVO, ARROZ, FRUTOSMAR, PORCO]
    #dominios = todos os valores que a variavel pode tomar
    dominio={SOPA:valoresSOPA[:],GALINHA:valoresGALINHA[:],VEGETAIS:valoresVEGETAIS[:],APERITIVO:valoresAPERITIVO[:],ARROZ:valoresARROZ[:],FRUTOSMAR:valoresFRUTOSMAR[:],PORCO:valoresPORCO[:]}
    
    #aqui coloco a resticao de cada prato.
    restricoes={SOPA:restricaoSOPA,GALINHA:restricaoGALINHA,VEGETAIS:restricaoVEGETAIS,APERITIVO:restricaoAPERITIVO,ARROZ:restricaoARROZ,FRUTOSMAR:restricaoFRUTOSMAR,PORCO:restricaoPORCO}
    
    
    return csp(variaveis,dominio,restricoes,n)


#################
#################
#################



def run():
 
    n="0"
    while not (int(n) > 1 and int(n) < 31):
        print " "
        n = raw_input("n? ")
        #print letra.upper()

    resultado =  BACKTRACKING_SEARCH(Chines(int(n)))  
    print "###############"
    print "## resultado ##"
    print "###############"
    if len(resultado)>1:
        for i in resultado:
            print resultado[i]
        print " "
        print "conta: " + str(Conta(resultado))   
    else:
        print "falha: impossivel atingir o menu!" 
        
        


run()
