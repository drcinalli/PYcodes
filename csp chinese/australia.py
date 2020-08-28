'''
Created on 17/04/2012

@author: quatrosem
'''

# classe para o CSP
class csp:
  def __init__(self, vars, domains, neighbors, constraints):
    ## these remain constant during search
    self.VARIABLES = vars
    self.DOMAINS = domains
    self.NEIGHBORS = neighbors
    self.CONSTRAINTS = constraints
#-----------------------------------------------------------

#chama programa para calcular resultado CSP
def BACKTRACKING_SEARCH(csp):
    return RECURSIVE_BACKTRACKING({},csp)


#CSP
def RECURSIVE_BACKTRACKING(assignment,csp):
    
    #se todo o conjunto de variaveis foi solucionada
    if COMPLETE(assignment,csp.VARIABLES):return assignment
    
    #buca prox variavel
    var=SELECT_UNASSIGNED_VARIABLE( csp.VARIABLES,assignment,csp)
    
    
    #para cada valor do dominio daquela variavel
    for value in ORDER_DOMAIN_VALUES(var,assignment,csp):
        
        #escolhido um valor
        #verifica se ele ficaria consistente com o restante dos assinalados
        if CONSISTENT(var,value,assignment,csp.CONSTRAINTS,csp.NEIGHBORS):
            #se pode essa cor, continua
            assignment[var]=value
            
            result=RECURSIVE_BACKTRACKING(assignment,csp)
            #volta da recursao
            if result!='failure':return result
            del assignment[var]
    
    #se nao fica consistente, ele volta mudando as cores dos antecessores.
    #esse algoritmo nao faz o forward checking....
    return 'failure'

#pega a proxima variavel da lista
def SELECT_UNASSIGNED_VARIABLE(vars,assignment,csp):
    for var in vars:
        if not (var in assignment):return var

#verifica se a lista de resultados esta completa  
def COMPLETE(assignment,vars):
    for var in vars:
        if not (var in assignment):return False
    return True

# Traz os dominios daquela variavel
def ORDER_DOMAIN_VALUES(var,assignment,csp):
    return csp.DOMAINS[var][:]

#verifica consistenca das respostas anteriores 
def CONSISTENT(var,value,assignment,constraints,neighbors):
    
    #se esta vazio, volta
    if not assignment:return True
    
    #circula por todas as restricoes
    for c in constraints.keys():
        #pega os vizinhos da variavel escolhida
        for var2 in neighbors[var]:
            #Se o vizinho estiver  na lista de resposta (esta pintado?) &
            # nao     no:cor novos + cor do pintado sao diferentes (se as cores dos 2 sao iguais, se forem retorna falsa e nao deixa pintar) 
            if var2 in assignment and not constraints[c](var,value,var2,assignment[var2]):
                return False
    return True        

## ele nao precisaria checar todo o conjunto de constraints ... mas acho q fez isso didaticamente
## pq em tese vc deveria repassar todas as restricoes e ver se estao ok.
## nesse caso especdico nao tem sentido pq ele olha apenas o novo no e os vizinhos.


#################
### Australia ###
#################
def Australia():#Return a CSP instance of the Map Coloring of Australia.
    WA,Q,T,V,SA,NT,NSW='WA','Q','T','V','SA','NT','NSW'
    #valor
    values=['RED','GREEN','BLUE']
    #variaveis
    vars=[WA,Q,T,V,SA,NT,NSW]
    #dominios = todos os valores que a variavel pode tomar
    domains={SA:values[:],WA:values[:],NT:values[:],Q:values[:],NSW:values[:],V:values[:],T:values[:]}
    #vizinhos
    neighbors={WA:[SA,NT],Q:[SA,NT,NSW],T:[],V:[SA,NSW],SA:[WA,NT,Q,NSW,V],NT:[SA,WA,Q],NSW:[SA,Q,V]}
    
    constraints={SA:constraint,WA:constraint,NT:constraint,Q:constraint,NSW:constraint,V:constraint,T:constraint}
    return csp(vars,domains,neighbors,constraints)

def constraint(A,a,B,b): #Two neighboring variables must differ in value.
    return a!=b

#################
#################
#################



def run():
    print BACKTRACKING_SEARCH(Australia())

run()
