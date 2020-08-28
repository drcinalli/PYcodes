# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0005_auto_20141108_2249'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='first_loop',
            field=models.IntegerField(default=2),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='population',
            name='chromosome_original',
            field=models.CharField(max_length=2000, null=True),
            preserve_default=True,
        ),
    ]
