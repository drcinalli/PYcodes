# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='player',
            name='objective1_pref',
            field=models.DecimalField(max_digits=4, decimal_places=2),
        ),
    ]
