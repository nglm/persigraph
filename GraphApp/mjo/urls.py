from django.urls import path
from . import views

urlpatterns = [
    path("", views.main),
    path("relevant/", views.relevant),
]