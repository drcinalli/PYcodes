# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0002_auto_20141030_2001'),
    ]

    operations = [
        migrations.AlterField(
            model_name='play',
            name='chromosomeOne',
            field=models.CharField(max_length=2000),
        ),
        migrations.AlterField(
            model_name='play',
            name='chromosomeTwo',
            field=models.CharField(max_length=2000),
        ),
        migrations.AlterField(
            model_name='player',
            name='email',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='player',
            name='name',
            field=models.CharField(max_length=200),
        ),
        migrations.AlterField(
            model_name='player',
            name='schooling',
            field=models.CharField(max_length=100),
        ),
    ]
