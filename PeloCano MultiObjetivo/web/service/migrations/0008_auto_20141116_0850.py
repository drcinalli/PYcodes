# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0007_auto_20141116_0840'),
    ]

    operations = [
        migrations.RenameField(
            model_name='generation',
            old_name='mean1',
            new_name='mean_1',
        ),
        migrations.RenameField(
            model_name='generation',
            old_name='mean2',
            new_name='mean_2',
        ),
        migrations.AddField(
            model_name='generation',
            name='p_1',
            field=models.FloatField(null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='generation',
            name='p_2',
            field=models.FloatField(null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='generation',
            name='sigma_1',
            field=models.FloatField(null=True),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='generation',
            name='sigma_2',
            field=models.FloatField(null=True),
            preserve_default=True,
        ),
    ]
