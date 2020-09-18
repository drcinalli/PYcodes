from django.contrib import admin
from service.models import Experiment, Generation, GameWorld, Area, Population, PFront, Play, Player


class GenerationInline(admin.TabularInline):
    model = Generation
    extra = 1


class ExperimentAdmin(admin.ModelAdmin):

    fieldsets = [
        (None,               {'fields': ['world']}),
        (None,               {'fields': ['name', 'description', 'type', 'type_prob', 'freeK','date', 'start']}),
        ('Experiment information', {'fields': ['flag','block_size','actual_gen', 'first_loop', 'gen_threshold',
                                               'num_robots','bots_points','numLevels','numMinPlayers', 'time_elapsed_end',
                                               'moea_alg', 'tour', 'vote',]}),
        ('Genetic information', {'fields': ['CXPB','MUTPB','NPOP', 'NGEN', 'paretoX_gen1', 'paretoY_gen1']}),
    ]
    inlines = [GenerationInline]
    list_display = ('name', 'flag', 'date', 'actual_gen')
    list_filter = ['date']
    search_fields = ['name']



class AreaInline(admin.TabularInline):
    model = Area
    extra = 1


class WorldAdmin(admin.ModelAdmin):
    fieldsets = [
        (None,               {'fields': ['name']}),
        ('World Information',               {'fields': ['m','n','max_areas','max_units']}),
        ('Production information', {'fields': ['prod_unit0','prod_unit1']}),
        ('Costs information', {'fields': ['cost_gateway','cost_unit0','cost_unit1']}),
        ('Cluster representation', {'fields': ['delta','x_line','y_line', 'cmap']}),
    ]
    inlines = [AreaInline]
    list_display = ('name', 'm', 'n')
    search_fields = ['name']


class PopulationInline(admin.TabularInline):
    model = Population
    extra = 1

class PFrontInline(admin.TabularInline):
    model = PFront
    extra = 1


class GenerationAdmin(admin.ModelAdmin):
    fieldsets = [
        (None,               {'fields': ['experiment', 'block', 'comparisons']}),
        ('Evolution Data',   {'fields': ['all_x', 'all_y', 'all_x_cp', 'all_y_cp', 'fitness_points2D']}),
        ('Cluster Data 1D',   {'fields': ['mean_1', 'mean_2','sigma_1', 'sigma_2', 'p_1', 'p_2']}),
        ('Cluster Data 2D',   {'fields': ['means', 'covar','weights', 'num_k']}),

    ]
    inlines = [PopulationInline, PFrontInline]
    list_display = ('experiment', 'block')
    search_fields = ['experiment']


class PlayInline(admin.TabularInline):
    model = Play
    extra = 0



class PlayerAdmin(admin.ModelAdmin):
    fieldsets = [
        (None,               {'fields': ['name', 'email', 'username', 'password', 'gender', 'age']}),
        ('Education Information',               {'fields': ['schooling']}),
        ('Robot Information',               {'fields': ['type', 'objective1_pref', 'f1_pref', 'f2_pref', 'f3_pref']}),

    ]
    inlines = [PlayInline]
    list_display = ('name', 'type', 'username')
    search_fields = ['name','username', 'email'  ]


class PlayAdmin(admin.ModelAdmin):

    fieldsets = [
        (None,               {'fields': ['play_experiment', 'play_player', 'level','chromosomeOne', 'chromosomeTwo' ]}),
    ]

    list_filter = ['play_experiment', 'level']




admin.site.register(Experiment, ExperimentAdmin)
admin.site.register(Generation, GenerationAdmin)
admin.site.register(GameWorld, WorldAdmin)
admin.site.register(Area)
admin.site.register(Population)
admin.site.register(PFront)
admin.site.register(Play, PlayAdmin)
admin.site.register(Player, PlayerAdmin)

