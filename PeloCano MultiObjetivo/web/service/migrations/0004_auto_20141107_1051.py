# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0003_auto_20141031_0252'),
    ]

    operations = [
        migrations.AlterField(
            model_name='experiment',
            name='type',
            field=models.CharField(max_length=1, choices=[(b'A', b'Only Gaussian Diff'), (b'B', b'COIN Elitism'), (b'C', b'Gaussian & COIN Elitism'), (b'Z', b'COIN Reference Point')]),
        ),
    ]
