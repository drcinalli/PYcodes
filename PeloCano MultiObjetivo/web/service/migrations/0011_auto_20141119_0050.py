# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0010_auto_20141119_0040'),
    ]

    operations = [
        migrations.AlterField(
            model_name='play',
            name='fit_custoOne',
            field=models.FloatField(default=-1),
        ),
        migrations.AlterField(
            model_name='play',
            name='fit_custoTwo',
            field=models.FloatField(default=-1),
        ),
        migrations.AlterField(
            model_name='play',
            name='fit_prodOne',
            field=models.FloatField(default=-1),
        ),
        migrations.AlterField(
            model_name='play',
            name='fit_prodTwo',
            field=models.FloatField(default=-1),
        ),
    ]
