from django.db import models

class InventoryItem(models.Model):
    CATEGORY_CHOICES = [
        ('Painkiller', 'Painkiller'),
        ('Antibiotic', 'Antibiotic'),
        ('Antiseptic', 'Antiseptic'),
        ('Supplement', 'Supplement'),
        ('Cough Relief', 'Cough Relief'),
        ('Equipment', 'Equipment'),
        ('First Aid', 'First Aid'),
    ]

    UNIT_CHOICES = [
    ('pcs', 'Pieces'),
    ('bottles', 'Bottles'),
    ('tabs', 'Tablets'),
    ('ml', 'Milliliters'),
    ('g', 'Grams'),
]

    STATUS_CHOICES = [
        ('In Stock', 'In Stock'),
        ('Low Stock', 'Low Stock'),
        ('Out of Stock', 'Out of Stock'),
    ]

    name = models.CharField(max_length=100)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default='Painkiller')
    quantity = models.PositiveIntegerField(default=0)
    unit = models.CharField(max_length=20, choices=UNIT_CHOICES, default='pcs')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='In Stock')
    date_added = models.DateField(auto_now_add=True)

    def __str__(self):
        return self.name
