from django.conf.urls import patterns, include, url
from . import views

urlpatterns = patterns('',
    url(r'^$', views.home),
    #url(r'^experiments/$', views.experiments),
)