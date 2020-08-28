# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('service', '0004_auto_20141107_1051'),
    ]

    operations = [
        migrations.AddField(
            model_name='experiment',
            name='paretoX_gen1',
            field=models.TextField(default=b''),
            preserve_default=True,
        ),
        migrations.AddField(
            model_name='experiment',
            name='paretoY_gen1',
            field=models.TextField(default=b''),
            preserve_default=True,
        ),
    ]
