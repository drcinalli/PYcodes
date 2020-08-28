// Agent fabrica_1 in project Bobinas.mas2j

/* Initial beliefs and rules */
pos(d,7).
qtde_bobina(1).

/* Initial goals */

!start.

/* Plans */

+!start : true <- 
				?qtde_bobina(A);
				.concat("Total de Bobinas:",A,X);
				.print(X).


//Incrementa Num. Bobinas
+incrementa[source(K)]<- 
						?qtde_bobina(Y);
						-+qtde_bobina(Y+1);
						.concat("Quantidade de Bobinas - de: ",Y,A);
						.concat(A,"  para:",B);
						.concat(B,Y+1,C);
						-incrementa[source(K)];
						.print(C).
						
//Decrementa Num. Bobinas

//se a proibicao existir, faz o mesmo, mas avisa caso norma tenha sido violada
//autonomia 100%, ele sempre vai burlar essa norma 
+decrementa[source(K)] : p(zero_bobinas) & qtde_bobina(Y) & Y < 2<- 
						//?qtde_bobina(Y);
						-+qtde_bobina(Y-1);
						.concat("Quantidade de Bobinas  (norma NAO atendida) - de: ",Y,A);
						.concat(A,"  para:",B);
						.concat(B,Y-1,C);
						-decrementa[source(K)];
				        .print(C);
						//avisa total ao juiz
						.send(juiz,tell,norma_linha_producao).
						
					
//se a proibicao existir, mas o numero de bobinas for acima de 1 ... faz a decrementacao (norma nao violada)
+decrementa[source(K)] : p(zero_bobinas) & qtde_bobina(Y) & Y > 1<- 
						//?qtde_bobina(Y);
						-+qtde_bobina(Y-1);
						.concat("Quantidade de Bobinas (norma atendida) - de: ",Y,A);
						.concat(A,"  para:",B);
						.concat(B,Y-1,C);
						-decrementa[source(K)];
				        .print(C).
						

//caso nao haja proibicao de zerar as bobinas						
+decrementa[source(K)] : not p(zero_bobinas)<- 
						?qtde_bobina(Y);
						-+qtde_bobina(Y-1);
						.concat("Quantidade de Bobinas - de: ",Y,A);
						.concat(A,"  para:",B);
						.concat(B,Y-1,C);
						-decrementa[source(K)];
				        .print(C).
				
											
