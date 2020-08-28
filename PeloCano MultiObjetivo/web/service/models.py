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

    def __unicode__(self):
        return self.name



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
    TYPE_EXP = (('A', 'Only Gaussian Diff'),
                ('B', 'COIN Elitism'),
                ('C', 'Gaussian & COIN Elitism'),
                ('Z', 'COIN Reference Point'),
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

    #name = models.CharField(max_length=110)
    type = models.CharField(max_length=1, choices=TYPE_EXP)
    description = models.CharField(max_length=1000)
    numLevels = models.IntegerField(default=5)
    numMinPlayers = models.IntegerField(default=7)
    start = models.DateTimeField()
    time_elapsed_end = models.BigIntegerField()

    CXPB = models.DecimalField(max_digits=2, decimal_places=2, default= 0.3)
    MUTPB= models.DecimalField(max_digits=2, decimal_places=2, default= 0.1)
    NGEN = models.IntegerField(default=100)
    NPOP = models.IntegerField(default=200)

    paretoX_gen1 = models.TextField(default="")
    paretoY_gen1 = models.TextField(default="")


    def __unicode__(self):
        return self.name

#generations table.
class Generation(models.Model):
    experiment = models.ForeignKey(Experiment)
    block = models.IntegerField(default=0)
    comparisons = models.CharField(max_length=2000, default="")
    all_x = models.TextField(default="")
    all_y = models.TextField(default="")
    all_x_cp = models.TextField(default="")
    all_y_cp = models.TextField(default="")
    mean_1 = models.FloatField(null=True)
    sigma_1 = models.FloatField(null=True)
    p_1 = models.FloatField(null=True)
    mean_2 = models.FloatField(null=True)
    sigma_2 = models.FloatField(null=True)
    p_2 = models.FloatField(null=True)


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
    fit_custoOne = models.FloatField(default=-1)
    fit_prodOne = models.FloatField(default=-1)
    fit_custoTwo = models.FloatField(default=-1)
    fit_prodTwo = models.FloatField(default=-1)

    def __unicode__(self):
        return str(self.answer)
