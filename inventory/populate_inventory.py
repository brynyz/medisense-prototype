import random
from inventory.models import InventoryItem
from django.utils import timezone

categories = ['Painkiller', 'Antibiotic', 'Antiseptic', 'Supplement', 'Cough Relief', 'Equipment', 'First Aid']
units = ['pcs', 'bottles', 'tabs', 'ml', 'g']
statuses = ['In Stock', 'Low Stock', 'Out of Stock']

sample_names = [
    'Paracetamol', 'Amoxicillin', 'Betadine', 'Vitamin C', 'Lagundi Syrup',
    'Thermometer', 'Bandage', 'Ibuprofen', 'Hydrogen Peroxide', 'Zinc Tablets'
]

for i in range(10):
    InventoryItem.objects.create(
        name=sample_names[i],
        category=random.choice(categories),
        quantity=random.randint(10, 200),
        unit=random.choice(units),
        status=random.choice(statuses),
        date_added=timezone.now()
    )
