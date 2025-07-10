import datetime
from django.http import HttpResponse
from openpyxl import Workbook
from xhtml2pdf import pisa
from django.template.loader import render_to_string
from django.shortcuts import render, redirect, get_object_or_404
from .models import InventoryItem
from django.views.decorators.csrf import csrf_exempt

def inventory_table(request):
    inventory_items = InventoryItem.objects.all().order_by('-date_added')
    return render(request, 'inventory/inventory_table.html', {
        'inventory_items': inventory_items,
        'category_choices': InventoryItem.CATEGORY_CHOICES,
        'unit_choices': InventoryItem.UNIT_CHOICES,
        'status_choices': InventoryItem.STATUS_CHOICES
    })

@csrf_exempt
def add_item(request):
    if request.method == 'POST':
        print("POST data received:", request.POST)

        InventoryItem.objects.create(
            name=request.POST.get('name'),
            category=request.POST.get('category'),
            quantity=int(request.POST.get('quantity')),
            unit=request.POST.get('unit'),
            status=request.POST.get('status')
        )
        return redirect('inventory_table')

    return redirect('inventory_table')


def edit_item(request, item_id):
    item = get_object_or_404(InventoryItem, id=item_id)

    if request.method == 'POST':
        item.name = request.POST.get('name')
        item.category = request.POST.get('category')
        item.quantity = request.POST.get('quantity')
        item.unit = request.POST.get('unit')
        item.status = request.POST.get('status')
        item.save()
        return redirect('inventory_table')

    inventory_items = InventoryItem.objects.all().order_by('-date_added')
    return render(request, 'inventory/inventory_table.html', {
        'inventory_items': inventory_items,
        'edit_item': item,
        'category_choices': InventoryItem.CATEGORY_CHOICES,
        'unit_choices': InventoryItem.UNIT_CHOICES,
        'status_choices': InventoryItem.STATUS_CHOICES
    })


def delete_item(request, item_id):
    item = get_object_or_404(InventoryItem, id=item_id)
    item.delete()
    return redirect('inventory_table')

def export_inventory_excel(request):
    wb = Workbook()
    ws = wb.active
    ws.title = "Inventory Items"

    # Headers
    ws.append(["Item Name", "Category", "Quantity", "Unit", "Status", "Date Added"])

    # Data
    for item in InventoryItem.objects.all():
        ws.append([
            item.name,
            item.category,
            item.quantity,
            item.get_unit_display(),
            item.status,
            item.date_added.strftime('%Y-%m-%d')
        ])

    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    filename = f"Inventory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    response['Content-Disposition'] = f'attachment; filename={filename}'
    wb.save(response)
    return response

def export_inventory_pdf(request):
    items = InventoryItem.objects.all()
    html_string = render_to_string('inventory/pdf_template.html', {'items': items})

    response = HttpResponse(content_type='application/pdf')
    filename = f"Inventory_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response['Content-Disposition'] = f'attachment; filename={filename}'

    pisa_status = pisa.CreatePDF(html_string, dest=response)
    if pisa_status.err:
        return HttpResponse("Error generating PDF", status=500)
    return response