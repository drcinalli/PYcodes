// Agent coordenador in project Bobinas.mas2j

/* Initial beliefs and rules */
sagarra.

/* Initial goals */

//!buscaLocalizacaoJumbos.
//!disponibilidadeJumbos.
!ini.

/* Plans */
//Intencao Primaria... verifica se existe programacao
//+!ini: true <-  !bobina;
+!ini: true <-  !disp;
				!ini.

//verifica se tem Jumbo Disponivel				
+!disp: not estado(Z,disponivel) <-				
			.wait(200);
			.print(" ... aguardando por programacao ...");
			//.random(H);
			//.print(H);
			!disponibilidadeJumbosSilent;
			.send(preactor,askOne,bobina(X,Y),A);
			+A.
			//.print(A);
			//.send(preactor,achieve,desmobilizar(X,Y)).
			
+!disp:  estado(Z,disponivel) & bobina(X,Y) & o(despachar)<-				
			!bobina.
			
+!disp:  estado(Z,disponivel) & bobina(X,Y) & not o(despachar)<-				
			.wait(200);
			.print(" ... aguardando por norma (programacao) ...").

			
+!disp:  estado(Z,disponivel) & not bobina(X,Y) <-				
			.wait(4000);
			.print(" ... fim da programacao ...");
			.wait(60000);
			.stopMAS.
			
			
//Caso tenha Bobina na fila de producao E disponibilidade dos jumbos
//coord. pede por suas localizacoes
+!bobina(X,Y): bobina(X,Y) & teste(A,disponivel) <- !buscaLocalizacaoJumbos;
											        .print(estado(A,disponivel));
													.print(bobina(X,Y));
													//aqui tenho disponibilidade de A e bobina(X,Y)
													//mando A transportar bobina
													//deleto fato A,disponivel
											        .wait(1000).
													//-disponivel(A).
													//q faco com a bobina?

//Caso tenha Bobina na fila de producao (preactor), o coordenador verifica se ha 
//disponibilidade dos jumbos e a posicao do jumbo disponivel.
+!bobina <-	//.send(preactor,askOne,bobina(X,Y),A);
			//+A;
			?bobina(X,Y)[source(H)];
			.concat("Bobina Despachada - DE:",X,I);
			.concat(I,"   PARA:",J);
			.concat(J,Y,K);			
			.print(K);
			//peco disponibilidade de alguem;
			//!disponibilidadeJumbos;
			//.wait();
			?estado(Z,disponivel);
			//mando pegar a posicao em que se encontra o Jumbo disponivel
			!buscaLocalizacaoJumboEspec(Z);
			//!buscaLocalizacaoJumbos;
			//mando levar
			//?bobina(N,M);
			.send(Z,achieve,levar(X,Y));
			//mando preactor limpar da sua base de programacoes
			.send(preactor,achieve,desmobilizar(X,Y));
			//limpo crenca da bobina programada
			-bobina(X,Y)[source(H)];
			//limpo disponibilidade ou untell
			!limpaDisponibilidades;
			//limpo posicoes ou untell
			!limpaPosicoes;
			//wait?
			//?pos(Z,Y) ;
			//.print(pos(Z,Y));
			.wait(1000).
			//proxima programacao
						
+!bobina <-	.wait(1000);
			.print(" ... aguardando disponibilidade ... ").


//broadcast para pegar disponibilidade
+!disponibilidadeJumbos:   true <-  
									.send(jumboA,askOne,estado(X,Y),A);
									+A;
									.send(jumboB,askOne,estado(X,Y),B);
									+B;
									.send(jumboC,askOne,estado(X,Y),C);
									+C;
									.print("Broadcast para disponibilidades dos Jumbos").

//broadcast para pegar disponibilidade
+!disponibilidadeJumbosSilent:   true <-  
									.send(jumboA,askOne,estado(X,Y),A);
									+A;
									.send(jumboB,askOne,estado(X,Y),B);
									+B;
									.send(jumboC,askOne,estado(X,Y),C);
									+C.
									

//broadcast para pegar posicionamento de um Jumbo especificamente
+!buscaLocalizacaoJumboEspec(X): estado(X,disponivel) <-	.send(X,askOne,pos(Z,Y),A);
															+A;
															.concat("Pede posi‹o do Jumbo: ",X,B);															
															.print(B).


//broadcast para pegar posicionamento dos jumbos
+!buscaLocalizacaoJumbos: true <-	.send(jumboA,askOne,pos(X,Y),A);
									+A;
									.send(jumboB,askOne,pos(X,Y),B);
									+B;
									.send(jumboC,askOne,pos(X,Y),C);
									+C;
									//.send(jumboA,tell,posicao);
  	 							  	//.send(jumboB,tell,posicao);
									//.send(jumboC,tell,posicao);
									.print("Broadcast para posicoes do Jumbo").

//faxina nas posicoes em Crencas
//+!limpaPosicoes: not pos(X,Y)  <- .print("Posicoes Limpas").
+!limpaPosicoes: not pos(X,Y)  <- true.
+!limpaPosicoes: pos(X,Y)  <- 	
								//?pos(X,Y)[source(A)] ;
								-pos(X,Y)[source(A)] ;
								//.print(xxxxxx);
								!limpaPosicoes.
								
//faxina nos estados de disponibilidades
//+!limpaDisponibilidades: not estado(X,Y) <- .print("Estados Limpos"). 	
+!limpaDisponibilidades: not estado(X,Y) <- true. 	
+!limpaDisponibilidades: estado(X,Y) <- 	
								//?pos(X,Y)[source(A)] ;
								-estado(X,Y)[source(A)] ;
								!limpaDisponibilidades.
								
