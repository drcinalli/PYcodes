

    #organize Gateways again
    def CheckGatewaysFIX(self, glist):
        if len(glist) != len(self.areas):
            print 'ERROR: different number of gateways and areas.'
            return 0

        #copy of areas, all orphan areas will be here
        cp_area  = list(self.areas)
        #gates with problem
        lst_gate = []

        #check each gateway
        j=0
        for i in glist:
            #if the gateway is not inside any area: keep its index!
            if self.CheckGate(i,cp_area)==0:
                lst_gate.append(j)
            j=j+1

        #here: orphan areas and wrong gateways
        j=0
        for i in cp_area:
            glist[lst_gate[j]] = self.CreateGate(i)
            j=j+1

        return 1

     #organize Units again
    def CheckUnitsFIX(self, ulist):
        if len(ulist) > self.all_units:
            print 'ERROR: superior number of units in domain.'
            return 0

        #copy of units, all orphan units will be here
        cp_units  = list(ulist)
        #units with problem
        lst_units = []

        #check each unit
        j=0
        for i in ulist:
            #if the unit is inside any area: keep its index!
            if self.CheckGate_onepass(i,self.areas)!=0:
                lst_units.append(j)
            j=j+1


        #here: wrong units
        j=0
        for i in lst_units:
            ulist[lst_units[j]] = self.CreateProdUnit(ulist)
            j=j+1

        return 1


    #organize Links again
    def CheckLinksFIX(self,glist,units, strategy):

        #cp_unit  = list(units)
        #cp_gates = list(glist)
        #list all not linked units
        lst_units=[]
        lst_units_final=[]
        #list the index of all the wrong gates
        lst_gates=[]
        units_dist =[]


        #check each link inside gateway
        j=0
        for i in glist:
            #if the gateway is not linked to some unit: get it on the list!
            if not (1<= i[2]<=len(units)):
                lst_gates.append(j)

            #if the unit is linked to some gateway, get it on the list
            else:
                if i[2] not in lst_units:
                    lst_units.append(i[2])

            j=j+1

        #list all not linked units
        for i in range(0,len(units)):
            if i+1 not in lst_units:
                lst_units_final.append(i+1)
                units_dist.append(units[i])

        #here, there are gates with mistake and non-orphan units in a list
        ##########################################################
        #STRATEGY: ? rand or closer gateways or better production#
        ##########################################################

        if strategy == 1:

            #wrong gates to all orphan units and not orphan
            #length orphan units
            len_unit = len(lst_units_final)
            #length wrong gates
            len_gates = len(lst_gates)
            for i in range(0,len_gates):
                #link gate to units available
                if len_unit > 1:
                    #rand a unit from the available
                    j = randint(0,len_unit-1)
                    glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],lst_units_final[j])
                    del lst_units_final[j]
                    len_unit = len_unit-1
                else:
                    #link to the unique unit available
                    if len_unit == 1:
                        #links to the unique unit
                        glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],lst_units_final[0])
                        del lst_units_final[0]
                        len_unit = len_unit-1
                    else:
                        # rand all units, because all units have been already linked
                        j = randint(1,len(units))
                        glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],j)


            #NO wrong gates and orphan units
            #orphan unit
            len_unit = len(lst_units_final)
            #original lenght of gateways
            len_gates = len(glist)

            #for each orphans
            for i in range(0,len_unit):
                #i is the index of a orphan unit
                # choose duplicated units in gateway list... the first ocurrences
                #c = Counter(elem[2] for elem in glist)
                duplicate=[]
                for k in range(0,len_gates):
                    for v in range(0,len_gates):
                        if v != k:
                            if glist[v][2]==glist[k][2]:
                                duplicate.append(v)
                                if k not in duplicate:
                                    duplicate.append(k)

                    #if a duplicate is found, get out
                    if duplicate:
                        break


                if duplicate:
                    aux = randint(0,len(duplicate)-1) ########
                    glist[duplicate[aux]]= (glist[duplicate[aux]][0],glist[duplicate[aux]][1],lst_units_final[i])
                else:
                    print 'ERROR: no duplication in gateways'
        elif strategy == 2:

            #wrong gates to all orphan units and not orphan
            #length orphan units
            len_unit = len(lst_units_final)
            #length wrong gates
            len_gates = len(lst_gates)
            for i in range(0,len_gates):
                #link gate to units available
                if len_unit > 1:
                    #get the shortest unit from the available
                    j = self.GetClosest(glist[i], units_dist)
                    glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],lst_units_final[j])
                    del lst_units_final[j]
                    del units_dist[j]
                    len_unit = len_unit-1
                else:
                    #link to the unique unit available
                    if len_unit == 1:
                        #links to the unique unit
                        glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],lst_units_final[0])
                        del lst_units_final[0]
                        len_unit = len_unit-1
                    else:
                        # get the shortest unit of all units, because all units have been already linked
                        j = self.GetClosest(glist[lst_gates[i]], units)
                        glist[lst_gates[i]] = (glist[lst_gates[i]][0],glist[lst_gates[i]][1],j+1)


            #NO wrong gates and orphan units
            #orphan unit
            len_unit = len(lst_units_final)
            #original lenght of gateways
            len_gates = len(glist)

            #for each orphans
            for i in range(0,len_unit):
                #i is the index of a orphan unit
                # choose duplicated units in gateway list... all ocurrences
                duplicate=[]
                for k in range(0,len_gates):
                    for v in range(0,len_gates):
                        if v != k:
                            if glist[v][2]==glist[k][2]:
                                if glist[v] not in duplicate:
                                    duplicate.append(glist[v])
                                if glist[k] not in duplicate:
                                    duplicate.append(glist[k])

                    #if a duplicate is found, get out
                    #if duplicate:
                        #break


                if duplicate:
                    #check unit and closest gateway... returns the index of the closest gateway (duplicate[aux])
                    aux=self.GetClosest(units[lst_units_final[i]-1], duplicate)
                    #aux = randint(0,len(duplicate)-1) ########
                    glist[glist.index(duplicate[aux])]= (glist[glist.index(duplicate[aux])][0], glist[glist.index(duplicate[aux])][1], lst_units_final[i])
                else:
                    print 'ERROR: no duplication in gateways'
