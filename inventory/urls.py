from django.urls import path
from . import views

urlpatterns = [
    path('', views.inventory_table, name='inventory_table'),
    path('add/', views.add_item, name='add_item'),
    path('edit/<int:item_id>/', views.edit_item, name='edit_item'),
    path('delete/<int:item_id>/', views.delete_item, name='delete_item'),
    path('export/excel/', views.export_inventory_excel, name='export_inventory_excel'),
    path('export/pdf/', views.export_inventory_pdf, name='export_inventory_pdf'),
]
