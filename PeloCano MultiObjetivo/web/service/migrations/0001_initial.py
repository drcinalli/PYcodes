# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Area',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('x', models.IntegerField()),
                ('y', models.IntegerField()),
                ('length', models.IntegerField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Experiment',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=200)),
                ('date', models.DateTimeField()),
                ('block_size', models.IntegerField(default=15)),
                ('flag', models.CharField(max_length=1, choices=[(b'W', b'Waiting'), (b'R', b'Ready'), (b'F', b'Finished'), (b'I', b'Idle')])),
                ('actual_gen', models.IntegerField(default=0)),
                ('gen_threshold', models.IntegerField(default=100)),
                ('num_robots', models.IntegerField(default=30)),
                ('type', models.CharField(max_length=1, choices=[(b'A', b'Experiment A'), (b'B', b'Experiment B'), (b'C', b'Experiment C')])),
                ('description', models.CharField(max_length=1000)),
                ('numLevels', models.IntegerField(default=5)),
                ('numMinPlayers', models.IntegerField(default=7)),
                ('start', models.DateTimeField()),
                ('time_elapsed_end', models.BigIntegerField()),
                ('CXPB', models.DecimalField(default=0.3, max_digits=2, decimal_places=2)),
                ('MUTPB', models.DecimalField(default=0.1, max_digits=2, decimal_places=2)),
                ('NGEN', models.IntegerField(default=100)),
                ('NPOP', models.IntegerField(default=200)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='GameWorld',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(default=b'Mundo 20x20', max_length=200)),
                ('m', models.IntegerField(default=20)),
                ('n', models.IntegerField(default=20)),
                ('max_areas', models.IntegerField(default=6)),
                ('max_units', models.IntegerField(default=6)),
                ('prod_unit0', models.IntegerField(default=23)),
                ('prod_unit1', models.IntegerField(default=35)),
                ('cost_gateway', models.IntegerField(default=5)),
                ('cost_unit0', models.IntegerField(default=13)),
                ('cost_unit1', models.IntegerField(default=17)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Generation',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('block', models.IntegerField(default=0)),
                ('comparisons', models.CharField(default=b'', max_length=2000)),
                ('experiment', models.ForeignKey(to='service.Experiment')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='PFront',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('chromosome', models.CharField(max_length=2000)),
                ('index', models.IntegerField(default=0)),
                ('generation', models.ForeignKey(to='service.Generation')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Play',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('level', models.IntegerField()),
                ('chromosomeOne', models.CharField(max_length=100)),
                ('chromosomeOneIndex', models.IntegerField()),
                ('chromosomeTwo', models.CharField(max_length=100)),
                ('chromosomeTwoIndex', models.IntegerField()),
                ('answer', models.IntegerField()),
                ('answer_time', models.BigIntegerField()),
                ('points', models.IntegerField()),
                ('play_experiment', models.ForeignKey(to='service.Experiment')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Player',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('username', models.CharField(max_length=30)),
                ('email', models.CharField(max_length=30)),
                ('password', models.CharField(max_length=50)),
                ('schooling', models.CharField(max_length=50)),
                ('gender', models.CharField(max_length=10)),
                ('age', models.IntegerField(max_length=3)),
                ('name', models.CharField(max_length=100)),
                ('type', models.CharField(max_length=1, choices=[(b'H', b'Human'), (b'C', b'Computer')])),
                ('objective1_pref', models.DecimalField(max_digits=2, decimal_places=2)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Population',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('chromosome', models.CharField(max_length=2000)),
                ('index', models.IntegerField(default=0)),
                ('generation', models.ForeignKey(to='service.Generation')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AddField(
            model_name='play',
            name='play_player',
            field=models.ForeignKey(to='service.Player'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='experiment',
            name='world',
            field=models.ForeignKey(to='service.GameWorld'),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='area',
            name='world',
            field=models.ForeignKey(to='service.GameWorld'),
            preserve_default=True,
        ),
    ]
