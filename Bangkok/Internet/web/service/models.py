from django.db import models

#generations table.
class GameWorld(models.Model):

    name = models.CharField(max_length=200,default='Mundo 20x20')

    #dimensions
    m=models.IntegerField(default=20)
    n=models.IntegerField(default=20)

    #max number of elements
    max_areas=models.IntegerField(default=6)
    max_units=models.IntegerField(default=6)

    #production of units type
    prod_unit0=models.IntegerField(default=23)
    prod_unit1=models.IntegerField(default=35)

    #costs
    cost_gateway=models.IntegerField(default=5)
    cost_unit0=models.IntegerField(default=13)
    cost_unit1=models.IntegerField(default=17)

    #cluster representation
    delta = models.FloatField(default=0.5)
    x_line = models.IntegerField(default=250)
    y_line = models.IntegerField(default=-250)
    cmap = models.CharField(max_length=100, default='Greens')


#areas in the World.
class Area(models.Model):
    world = models.ForeignKey(GameWorld)
    x = models.IntegerField()
    y = models.IntegerField()
    length = models.IntegerField()


#experiments table.
class Experiment(models.Model):

    FLAGS = (('W', 'Waiting'),
             ('R', 'Ready'),
             ('F', 'Finished'),
             ('I', 'Idle')
    )
    TYPE_EXP = (('A', '1-D Gaussian Mixture'),
                ('B', '2-D Gaussian Mixture'),
                ('C', '2-D K-means'),
                ('D', '3-D Gaussian Mixture'),
    )

    TYPE_PROBLEM = (('A', 'Resource Distribution'),
                    ('F', 'DTLZ-1'),
                    ('G', 'DTLZ-2'),
                    ('H', 'DTLZ-3'),
                    ('I', 'DTLZ-4'),
                    ('J', 'DTLZ-5'),
                    ('L', 'DTLZ-6'),
                    ('M', 'DTLZ-7'),
                    ('P', 'ZDT-1'),
                    ('Q', 'ZDT-2'),
                    ('R', 'ZDT-3'),
                    ('S', 'ZDT-4'),
                    ('T', 'ZDT-6'),
    )

    #inform where the Bots will vote ... and give a hint to the number of K if freek is False
    BOTS_points = (('A', 'Middle'),
                   ('B', '2-points in the extreme'),
                   ('C', 'Origin and 2-points in the extreme'),
                   ('D', 'N-points random'),
    )

    #inform the MOEA algorithm
    MOEA_ALG    = (('N', 'COIN.NSGA-II'),
                   ('S', 'COIN.SMS-EMOA'),
                   ('P', 'COIN.SMS-EMOA'),
    )

    #inform the sort of Tornment in NSGA-II (so far)
    TOUR        = (('R', 'NSGA-II regular'),
                   ('C', 'NSGA-II cluster diversity FULL'),
                   ('A', 'NSGA-II cluster diversity ONE'),
    )

    #inform if Vote will be on P* or all Pop
    VOTE        = (('P', 'Pareto Front'),
                   ('A', 'All population'),
    )
    world = models.ForeignKey(GameWorld)

    name = models.CharField(max_length=200)
    date = models.DateTimeField()
    block_size= models.IntegerField(default=15)
    flag = models.CharField(max_length=1, choices=FLAGS)
    actual_gen= models.IntegerField(default=0)
    gen_threshold=models.IntegerField(default=100)
    first_loop=models.IntegerField(default=2)
    num_robots=models.IntegerField(default=30)

    type_prob = models.CharField(max_length=1, choices=TYPE_PROBLEM, default='A')


    #name = models.CharField(max_length=110)
    type = models.CharField(max_length=1, choices=TYPE_EXP)
    freeK= models.BooleanField(default=False)

    moea_alg = models.CharField(max_length=1, choices=MOEA_ALG, default='N')
    tour     = models.CharField(max_length=1, choices=TOUR, default='R')
    vote     = models.CharField(max_length=1, choices=VOTE, default='P')

    description = models.CharField(max_length=1000)
    numLevels = models.IntegerField(default=5)
    numMinPlayers = models.IntegerField(default=7)
    start = models.DateTimeField()
    time_elapsed_end = models.BigIntegerField()

    bots_points = models.CharField(max_length=1, choices=BOTS_points, default='A')


    CXPB = models.DecimalField(max_digits=2, decimal_places=2, default= 0.3)
    MUTPB= models.DecimalField(max_digits=2, decimal_places=2, default= 0.1)
    NGEN = models.IntegerField(default=100)
    NPOP = models.IntegerField(default=200)

    paretoX_gen1 = models.TextField(default=" ")
    paretoY_gen1 = models.TextField(default=" ")


    def __unicode__(self):
        return self.name

#generations table.
class Generation(models.Model):
    experiment = models.ForeignKey(Experiment)
    block = models.IntegerField(default=0)
    comparisons = models.CharField(max_length=2000, default="")
    all_x = models.TextField(default=" ")
    all_y = models.TextField(default=" ")
    all_x_cp = models.TextField(default=" ")
    all_y_cp = models.TextField(default=" ")
    #this is for the 1-D gaussian experiment
    mean_1 = models.FloatField(null=True,default=0)
    sigma_1 = models.FloatField(null=True,default=0)
    p_1 = models.FloatField(null=True,default=0)
    mean_2 = models.FloatField(null=True,default=0)
    sigma_2 = models.FloatField(null=True,default=0)
    p_2 = models.FloatField(null=True,default=0)
    #this is for the Multivariate Gaussian
    means = models.CharField(max_length=1000, default=" ")
    covar = models.CharField(max_length=1000, default=" ")
    weights = models.CharField(max_length=1000, default=" ")
    #this is the points chosen ... need this to the 2D gaussian page
    fitness_points2D = models.TextField(default=" ")
    num_k = models.IntegerField(default=0)
    #kmeans centroids
    centroids = models.TextField(default=" ")




    def __unicode__(self):
        return str(self.block)

#individuals table.
class Population(models.Model):
    generation = models.ForeignKey(Generation)
    chromosome = models.CharField(max_length=2000)
    index = models.IntegerField(default=0)
    chromosome_original = models.CharField(max_length=2000, null=True)


    def __unicode__(self):
        return self.chromosome



#individuals table.
class PFront(models.Model):
    generation = models.ForeignKey(Generation)
    chromosome = models.CharField(max_length=2000)
    index = models.IntegerField(default=0)
    #chromosome_original = models.CharField(max_length=2000, null=True)

    def __unicode__(self):
        return self.chromosome

class Player(models.Model):
    TYPE_PLAYER = (('H', 'Human'),
                   ('C', 'Computer'),
    )


    username = models.CharField(max_length=30)
    email = models.CharField(max_length=100)
    password = models.CharField(max_length=50)
    schooling = models.CharField(max_length=100)
    gender = models.CharField(max_length=10)
    age = models.IntegerField(max_length=3)
    name = models.CharField(max_length=200)
    type = models.CharField(max_length=1, choices=TYPE_PLAYER)
    objective1_pref = models.DecimalField(max_digits=4, decimal_places=2)
    f1_pref = models.FloatField(default=0)
    f2_pref = models.FloatField(default=0)
    f3_pref = models.FloatField(default=0)

    def __unicode__(self):
        return self.name



#class Experiment(models.Model):
#    name = models.CharField(max_length=110)
#    type = models.CharField(max_length=110)
#    description = models.CharField(max_length=100)
#    numLevels = models.IntegerField()
#    numMinPlayers = models.IntegerField()
#    start = models.DateTimeField()
#    time_elapsed_end = models.BigIntegerField()


class Play(models.Model):
    level = models.IntegerField()
    chromosomeOne = models.CharField(max_length=2000)
    chromosomeOneIndex = models.IntegerField()
    chromosomeTwo = models.CharField(max_length=2000)
    chromosomeTwoIndex = models.IntegerField()
    answer = models.IntegerField()
    answer_time = models.BigIntegerField()
    points = models.IntegerField()
    play_player = models.ForeignKey(Player)
    play_experiment = models.ForeignKey(Experiment)

    def __unicode__(self):
        return str(self.answer)
