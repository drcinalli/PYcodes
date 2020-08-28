from django.shortcuts import render

# Create your views here.
def home(request):
    return render(request, 'administration/home.html', {})

def experiments(request):
    return render(request, 'administration/experiments.html', {})

def interfaces(request):
    return render(request, 'administration/interfaces.html', {})

def game(request):
    return render(request, 'administration/Jogo-V3.html', {})