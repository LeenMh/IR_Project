# Generated by Django 4.0.2 on 2022-05-21 22:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('base', '0001_initial'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='dataset1',
            name='author',
        ),
        migrations.RemoveField(
            model_name='dataset1',
            name='title',
        ),
    ]