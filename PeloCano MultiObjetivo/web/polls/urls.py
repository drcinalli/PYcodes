from django.conf.urls import url

from polls import views

urlpatterns = [
    # ex: /polls/
    url(r'^$', views.index, name='index'),
    # ex: /polls/5/
    url(r'^(?P<exp_id>[0-9]+)/$', views.detail, name='detail'),
    # ex: /polls/5/results/
    url(r'^(?P<exp_id>[0-9]+)/results/$', views.results, name='results'),
    # ex: /polls/5/vote/
    url(r'^(?P<exp_id>[0-9]+)/vote/$', views.vote, name='vote'),
    # ex: /polls/start/
    url(r'start/$', views.start, name='start'),
    # ex: /polls/new/
    url(r'new/$', views.new, name='new'),
    # ex: /polls/start/go/
    url(r'^(?P<exp_id>[0-9]+)/go/$', views.go, name='go'),
    # ex: /polls/5/error/
    url(r'^(?P<exp_id>[0-9]+)/error/$', views.error, name='error'),
    # ex: /polls/5/getresults/
    url(r'getresults/$', views.getresults, name='getresults'),
    # ex: /polls/5/setresults/
    url(r'setresults/$', views.setresults, name='setresults'),
    # ex: /polls/5/setresults/
    url(r'^gauss/$', views.gauss, name='gauss'),

    ]
