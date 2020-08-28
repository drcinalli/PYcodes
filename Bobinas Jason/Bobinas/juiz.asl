// Agent juiz in project Bobinas.mas2j

/* Initial beliefs and rules */

/* Initial goals */

!ini.

/* Plans */

+!ini : true <- 
				//distribuicao de normas
			    .send(coordenador,tell,o(despachar));
			    .send(jumboA,tell,o(entregar_local_correto));
			    .send(jumboB,tell,o(entregar_local_correto));
			    .send(jumboC,tell,o(entregar_local_correto));
			    .send(fabrica_1,tell,p(zero_bobinas));
			    .send(fabrica_2,tell,p(zero_bobinas)).
				 

//recebe mensagens e aplica SANCOES ou GOVERNANCAS
+norma_entrega(X)[source(A)]<- 
			    .send(X,tell,punir_entrega);
				.concat("Sancao imposta: ",X,P);
				.print(P);
				-norma_entrega(X)[source(A)].

+norma_linha_producao[source(A)]<- 
			    .send(preactor,tell,reprogramar(A));
				.concat("Governanca imposta: ",A,P);
				.print(P);
				-norma_linha_producao[source(A)].

				

//age com punicoes
