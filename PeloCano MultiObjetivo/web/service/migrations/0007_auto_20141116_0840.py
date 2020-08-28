# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0006_auto_20141115_0937'),
    ]

    operations = [
        migrations.AddField(
            model_name='generation',
            name='all_x',
            field=models.TextField(default=b''),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='generation',
            name='all_x_cp',
            field=models.TextField(default=b''),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='generation',
            name='all_y',
            field=models.TextField(default=b''),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='generation',
            name='all_y_cp',
            field=models.TextField(default=b''),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='generation',
            name='mean1',
            field=models.FloatField(null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='generation',
            name='mean2',
            field=models.FloatField(null=True),
            preserve_default=True,
        ),
    ]
