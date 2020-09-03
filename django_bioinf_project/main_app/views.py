from django.http import HttpResponse
from django.shortcuts import render
from project_settings.settings import STATIC_URL


def index(request):
    return HttpResponse("Привет, мир!")

def self_input_form(request):
    if request.POST:
        print("Принял!")
    res = HttpResponse()
    res.status_code = 200
    res['Access-Control-Allow-Origin'] = 'http://127.0.0.1:8080'
    res['Access-Control-Allow-Credentials'] = 'true'
    return res
