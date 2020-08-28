// Agent Preactor in project Bobinas.mas2j

/* Initial beliefs and rules */
/* Initial beliefs and rules */
bobina(fabrica_1,patio_2).
bobina(fabrica_1,patio_1).
bobina(porto,fabrica_2).
bobina(porto,patio_2).
bobina(patio_1,patio_2).
bobina(porto,patio_3).
bobina(patio_3,fabrica_1).
bobina(fabrica_2, patio_2).
bobina(fabrica_1,fabrica_2).
bobina(porto,patio_1).
bobina(porto,patio_3).
bobina(patio_3,fabrica_1).
bobina(patio_2,porto).
bobina(patio_1,porto).
bobina(patio_3,patio_2).
bobina(porto,fabrica_1).
bobina(patio_3,fabrica_2).
bobina(fabrica_1, patio_1).
bobina(fabrica_2,fabrica_1).
bobina(porto,patio_3).
bobina(porto,patio_1).
bobina(patio_2,fabrica_2).
bobina(patio_1,porto).


/* Initial goals */

//!programacao.

/* Plans */

//envia programacao das bobinas
+!programacao : bobina(X,Y) <-?bobina(X,Y); 
				.send(coordenador,tell,bobina(X,Y));
				.wait(2000);
				-bobina(X,Y);
				!programacao.
				
+!programacao <-.print("Fim da Programacao"). 

//desmobiliza a programacao j� enviada ao Jumbo
+!desmobilizar(N,M)[source(A)]  <-
									//.print(xxxxxxxxx);
									-bobina(N,M) . 

+reprogramar(M)[source(A)]  <-
								//.print(xxxxxxxxx);
								?bobina(X,Y);
								-bobina(X,Y);
								+bobina(X,M);
								.concat("GOVERNANCA: reprogramacao de ", Y, B);
								.concat(B, "  para ", M, C);
								
								.print(C);
								//.print(X);
								-reprogramar(M)[source(A)]. 


