// Agent jumboC in project Bobinas.mas2j

/* Initial beliefs and rules */
pos(f,8).
//seu estado: disponivel ou indisponivel
estado(jumboC, disponivel).
//meta Jumbo
meta(0).


/* Initial goals */


/* Plans */
//reporta a sua posicao no mapa
+posicao[source(A)] 
  <- ?pos(X,Y);
     .send(A,tell,pos(X,Y));
	 .print(pos(X,Y));
	 -posicao[source(A)].

//reporta seu estado
+disponibilidade[source(A)] 
  <- ?disponivel(X);
  	 .send(A,tell,disponivel(X));
	 .print(disponivel(X));
	 -disponibilidade[source(A)].
	 
	 
//ir para o ponto ORIGEM
+!ir(X) <-
			?pos(Z,Y)[source(A)];
			-pos(Z,Y)[source(A)];
			+X;
			.concat("Estou em (x,y): ",X,I);
			.print(I).

//levar bobina de origem para destino
+!levar(X,Y) <- 
				//update no status
				-+estado(jumboC,indisponivel);
				//pegar posicao X
				.send(X,askOne,pos(I,J),A);
				//+A;
				//pegar posicao Y
				.send(Y,askOne,pos(K,L),B);
				//+B;
				//ir X
				!ir(A);
				//wait - tempo pra ir
				.wait(2000);
				//pedir bobina (decrementa numero bobinas)
				.send(X,tell,decrementa);
				//ir Y
				!ir(B);
				//wait - tempo pra ir
				.wait(2000);
				!entregar_bobina(X,Y).
				

//entregar bobina no destino
+!entregar_bobina(X,Y): true <- 
				//entregar bobina 
				!entregar(X,Y).

//40% de chance para NAO fazer a entrega ap—s meta alcanada.
+!entregar(X,Y): o(entregar_local_correto) & meta(K) & K>= 80 & .random(H) & H < 0.4 <- 
				//NAO entregar bobina (NAO incrementa numero bobinas)
				//NAO entrega bobina (NAO manda bobina para o local)
				//NAO Atualiza meta
				//muda seu status
				-+estado(jumboC,disponivel);
				.send(juiz,tell,norma_entrega(jumboC));
				.print("Bobina NAO Entregue (norma nao atendida)!").
				//atualiza sua posicao
				//posicao ja atualizada

				
//meta alcancada e rand para FUGA nao atuado, por isso abaixo ele faz a entrega
+!entregar(X,Y): o(entregar_local_correto) & meta(K) & K>= 80 <- 
				//entregar bobina (incrementa numero bobinas)
				.send(Y,tell,incrementa);
				//entrega bobina (manda bobina para o local)
				.send(Y,tell,bobina(X,Y));
				//Atualiza meta
				?meta(M);
				-+meta(M+20);
				//muda seu status
				-+estado(jumboC,disponivel);
				.concat("Meta: ",M+20,P);
				.print(P);
				.print("Bobina Entregue (norma atendida)!").
				//atualiza sua posicao
				//posicao ja atualizada
	
//Se abaixo da meta, nao questiona a norma
+!entregar(X,Y): o(entregar_local_correto) & meta(K) & K< 80 <- 
				//entregar bobina (incrementa numero bobinas)
				.send(Y,tell,incrementa);
				//entrega bobina (manda bobina para o local)
				.send(Y,tell,bobina(X,Y));
				//Atualiza meta
				?meta(M);
				-+meta(M+20);
				//muda seu status
				-+estado(jumboC,disponivel);
				.concat("Meta: ",M+20,P);
				.print(P);
				.print("Bobina Entregue!").
				//atualiza sua posicao
				//posicao ja atualizada

//sem a Norma de Obrigacao, ele nao faz nada
+!entregar(X,Y): not o(entregar_local_correto) <- 
				//NAO entregar bobina (NAO incrementa numero bobinas)
				//NAO entrega bobina (NAO manda bobina para o local)
				//NAO Atualiza meta
				//muda seu status
				-+estado(jumboC,disponivel);
				.print("Bobina NAO Entregue!").
				//atualiza sua posicao
				//posicao ja atualizada

//SANCAO para quebra de normas
+punir_entrega[source(A)]: true <-
					?meta(M);
					-+meta(M-15);
					.concat("Meta (punicao): ",M-15,P);
					.print(P);
					-punir_entrega[source(A)].
					



				
