from django.urls import path
from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('send-self-input-form', views.self_input_form, name='self_input_form')
]