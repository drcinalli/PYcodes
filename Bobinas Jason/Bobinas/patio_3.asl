// Agent patio_3 in project Bobinas.mas2j

/* Initial beliefs and rules */
pos(e,19).
qtde_bobina(100).

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
+decrementa[source(K)] <- 
						?qtde_bobina(Y);
						-+qtde_bobina(Y-1);
						.concat("Quantidade de Bobinas - de: ",Y,A);
						.concat(A,"  para:",B);
						.concat(B,Y-1,C);
						-decrementa[source(K)];
				        .print(C).
						
						
