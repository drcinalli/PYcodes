# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0008_auto_20141116_0850'),
    ]

    operations = [
        migrations.AddField(
            model_name='play',
            name='fit_custoOne',
            field=models.DecimalField(default=-1, max_digits=4, decimal_places=2),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='play',
            name='fit_custoTwo',
            field=models.DecimalField(default=-1, max_digits=4, decimal_places=2),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='play',
            name='fit_prodOne',
            field=models.DecimalField(default=-1, max_digits=4, decimal_places=2),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='play',
            name='fit_prodTwo',
            field=models.DecimalField(default=-1, max_digits=4, decimal_places=2),
            preserve_default=False,
        ),
    ]
