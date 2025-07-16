from django.core.management.base import BaseCommand
from inventory.populate_inventory import populate_inventory

class Command(BaseCommand):
    help = 'Populate the inventory with sample medical supplies data'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing inventory before populating',
        )

    def handle(self, *args, **options):
        self.stdout.write(
            self.style.SUCCESS('Starting inventory population...')
        )
        
        try:
            populate_inventory()
            self.stdout.write(
                self.style.SUCCESS('✅ Inventory populated successfully!')
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f'❌ Error populating inventory: {e}')
            )
