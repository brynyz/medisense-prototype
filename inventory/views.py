import datetime
from django.http import HttpResponse, JsonResponse
from openpyxl import Workbook
from xhtml2pdf import pisa
from django.template.loader import render_to_string
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from .models import InventoryItem
from django.views.decorators.csrf import csrf_exempt

@login_required
def inventory_table(request):
    inventory_items = InventoryItem.objects.all().order_by('-date_added')
    return render(request, 'inventory/inventory_table.html', {
        'inventory_items': inventory_items,
        'category_choices': InventoryItem.CATEGORY_CHOICES,
        'unit_choices': InventoryItem.UNIT_CHOICES,
        'status_choices': InventoryItem.STATUS_CHOICES
    })

@login_required
def add_item(request):
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            category = request.POST.get('category')
            quantity = request.POST.get('quantity')
            unit = request.POST.get('unit')
            status = request.POST.get('status')
            
            print(f"DEBUG - Received data: name={name}, category={category}, quantity={quantity}, unit={unit}, status={status}")
            
            # Validate required fields
            if not name:
                return JsonResponse({'success': False, 'error': 'Item name is required'})
            if not category:
                return JsonResponse({'success': False, 'error': 'Category is required'})
            if not quantity:
                return JsonResponse({'success': False, 'error': 'Quantity is required'})
            if not unit:
                return JsonResponse({'success': False, 'error': 'Unit is required'})
            if not status:
                return JsonResponse({'success': False, 'error': 'Status is required'})
            
            # Create the item
            new_item = InventoryItem.objects.create(
                name=name,
                category=category,
                quantity=int(quantity),
                unit=unit,
                status=status,
                last_modified_by=request.user
            )
            
            return JsonResponse({
                'success': True,
                'message': 'Item added successfully!',
                'item_data': {
                    'id': new_item.id,
                    'name': new_item.name,
                    'category': new_item.category,
                    'quantity': new_item.quantity,
                    'unit': new_item.get_unit_display(),
                    'status': new_item.status,
                    'date_added': new_item.date_added.strftime('%Y-%m-%d')
                }
            })
            
        except ValueError as e:
            return JsonResponse({'success': False, 'error': 'Invalid quantity - must be a number'})
        except Exception as e:
            print(f"DEBUG - Error creating item: {str(e)}")
            return JsonResponse({'success': False, 'error': f'Error creating item: {str(e)}'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required 
def edit_item(request, item_id):
    item = get_object_or_404(InventoryItem, id=item_id)
    
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            category = request.POST.get('category')
            quantity = request.POST.get('quantity')
            unit = request.POST.get('unit')
            status = request.POST.get('status')
            
            # Validate required fields
            if not name:
                return JsonResponse({'success': False, 'error': 'Item name is required'})
            if not category:
                return JsonResponse({'success': False, 'error': 'Category is required'})
            if not quantity:
                return JsonResponse({'success': False, 'error': 'Quantity is required'})
            if not unit:
                return JsonResponse({'success': False, 'error': 'Unit is required'})
            if not status:
                return JsonResponse({'success': False, 'error': 'Status is required'})
            
            # Update the item
            item.name = name
            item.category = category
            item.quantity = int(quantity)
            item.unit = unit
            item.status = status
            item.last_modified_by = request.user
            item.save()
            
            return JsonResponse({
                'success': True,
                'message': 'Item updated successfully!',
                'item_data': {
                    'id': item.id,
                    'name': item.name,
                    'category': item.category,
                    'quantity': item.quantity,
                    'unit': item.get_unit_display(),
                    'status': item.status,
                    'date_added': item.date_added.strftime('%Y-%m-%d')
                }
            })
            
        except ValueError as e:
            return JsonResponse({'success': False, 'error': 'Invalid quantity - must be a number'})
        except Exception as e:
            print(f"DEBUG - Error updating item: {str(e)}")
            return JsonResponse({'success': False, 'error': f'Error updating item: {str(e)}'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

@login_required
def delete_item(request, item_id):
    if request.method == 'POST':
        try:
            item = get_object_or_404(InventoryItem, id=item_id)
            item.delete()
            
            return JsonResponse({
                'success': True,
                'message': 'Item deleted successfully!'
            })
            
        except Exception as e:
            print(f"DEBUG - Error deleting item: {str(e)}")
            return JsonResponse({'success': False, 'error': f'Error deleting item: {str(e)}'})
    
    return JsonResponse({'success': False, 'error': 'Invalid request method'})

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
    items = InventoryItem.objects.all().order_by('category', 'name')
    
    # Calculate statistics
    total_items = items.count()
    total_quantity = sum(item.quantity for item in items)
    
    # Items per category
    from django.db.models import Count, Sum
    category_stats = items.values('category').annotate(
        count=Count('id'),
        total_qty=Sum('quantity')
    ).order_by('category')
    
    # Items per status
    status_stats = items.values('status').annotate(
        count=Count('id'),
        total_qty=Sum('quantity')
    ).order_by('status')
    
    # Items per unit
    unit_stats = items.values('unit').annotate(
        count=Count('id'),
        total_qty=Sum('quantity')
    ).order_by('unit')
    
    # Low stock items (assuming "Low Stock" or quantity < 10)
    low_stock_items = items.filter(status='Low Stock')
    critical_items = items.filter(quantity__lt=5)  # Items with less than 5 units
    
    # Recent additions (last 30 days)
    from datetime import datetime, timedelta
    thirty_days_ago = datetime.now().date() - timedelta(days=30)
    recent_items = items.filter(date_added__gte=thirty_days_ago)
    
    # Most stocked items (top 10)
    top_stocked = items.order_by('-quantity')[:10]
    
    # Prepare context for PDF template
    context = {
        'items': items,
        'report_date': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
        'total_items': total_items,
        'total_quantity': total_quantity,
        'category_stats': category_stats,
        'status_stats': status_stats,
        'unit_stats': unit_stats,
        'low_stock_items': low_stock_items,
        'critical_items': critical_items,
        'recent_items': recent_items,
        'top_stocked': top_stocked,
        'low_stock_count': low_stock_items.count(),
        'critical_count': critical_items.count(),
        'recent_count': recent_items.count(),
    }
    
    html_string = render_to_string('inventory/pdf_report_template.html', context)

    response = HttpResponse(content_type='application/pdf')
    filename = f"Inventory_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    response['Content-Disposition'] = f'attachment; filename={filename}'

    pisa_status = pisa.CreatePDF(html_string, dest=response)
    if pisa_status.err:
        return HttpResponse("Error generating PDF", status=500)
    return response

