# Generated by Django 5.2.3 on 2025-07-12 16:27

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("accounts", "0001_initial"),
    ]

    operations = [
        migrations.DeleteModel(
            name="User",
        ),
    ]
