from django.http import HttpResponse
from django.shortcuts import render
from project_settings.settings import STATIC_URL


def index(request):
    #return render(request, "main_app/index.html")
    return HttpResponse("Привет, мир!")

def self_input_form(request):
    if request.POST:
        print("Принял!")
