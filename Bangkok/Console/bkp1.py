   #TEST: create full individual
    for i in range(0,1):
        ind = my_world.CreateFull()
        print ind



        #gates=my_world.GetGates(ind)
        #units=my_world.GetUnits(ind)
        #Plot the World
        #my_world.PlotWorldDetails(gates,units)

        toolbox = base.Toolbox()

        #parameters: mutation and number of generations
        CXPB, MUTPB, NGEN = 0.5, 0.2, 40
        num_population    = 50


        #define fitness of the individual
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)


        # Attribute generator
        #toolbox.register("individual_gen", CreateFull)
        # Attribute generator
        #toolbox.register("attr_bool", random.randint, 0, 1)

        #def testezao():
        #    return [8, 8, 1, 19, 15, 1, 16, 11, 1, 14, 2, 1, 4, 15, 1, 9, 14, 1, 0, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1]


        # Attribute generator
        #toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual_gen", my_world.CreateFull)


        # Structure initializers
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.individual_gen, 1)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)


        #Fitness of OBJECTIVE #1
        def f1Cost(individual):

            return sum(individual)

        # Operator registering
        toolbox.register("evaluate", f1Cost)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        pop = toolbox.population(n=3)
        print pop
        pop2=[]
        for i in pop:
            pop2.append(i[0])
        print pop2

        fitnesses = list(map(toolbox.evaluate, pop))
        #fitnesses = list(map(toolbox.evaluate, [[0, 1, 0], [1, 0, 1], [1, 1, 1]]))
        print fitnesses
