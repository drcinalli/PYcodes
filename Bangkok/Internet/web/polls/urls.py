from django.conf.urls import url

from polls import views

urlpatterns = [
    # ex: /polls/
    # url(r'^$', views.index, name='index'),
    # ex: /polls/5/
    # url(r'^(?P<exp_id>[0-9]+)/$', views.detail, name='detail'),
    # ex: /polls/5/results/
    # url(r'^(?P<exp_id>[0-9]+)/results/$', views.results, name='results'),
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
    #
    url(r'^gauss/$', views.gauss, name='gauss'),
    #
    url(r'^bigausscontour/$', views.bigausscontour, name='bigausscontour'),
    url(r'^bigaussbell/$', views.bigaussbell, name='bigaussbell'),
    url(r'^bigausscontour_b/$', views.bigausscontour_b, name='bigausscontour_b'),
    url(r'^bigaussbell_b/$', views.bigaussbell_b, name='bigaussbell_b'),
    url(r'^kmeans/$', views.kmeans, name='kmeans'),
    url(r'^kmeans_b/$', views.kmeans_b, name='kmeans_b'),
    url(r'^finalfront_gauss/$', views.finalfront_gauss, name='finalfront_gauss'),
    url(r'^finalfront_gauss_b/$', views.finalfront_gauss_b, name='finalfront_gauss_b'),
    url(r'^finalfront_kmeans/$', views.finalfront_kmeans, name='finalfront_kmeans'),
    url(r'^finalfront_kmeans_b/$', views.finalfront_kmeans_b, name='finalfront_kmeans_b'),

    url(r'^finalfront_gauss_line/$', views.finalfront_gauss_line, name='finalfront_gauss_line'),
    url(r'^finalfront_gauss_line_b/$', views.finalfront_gauss_line_b, name='finalfront_gauss_line_b'),
    url(r'^finalfront_kmeans_line/$', views.finalfront_kmeans_line, name='finalfront_kmeans_line'),
    url(r'^finalfront_kmeans_line_b/$', views.finalfront_kmeans_line_b, name='finalfront_kmeans_line_b'),


    url(r'^humanvote/$', views.humanvote, name='humanvote'),


    # ex: /polls/home/
    url(r'home/$', views.home, name='home'),
    # ex: /polls/articles/
    url(r'articles/$', views.articles, name='articles'),


    # ex: /polls/startBench/
    url(r'startBench/$', views.startBench, name='startBench'),
    # ex: /polls/newBench/
    url(r'newBench/$', views.newBench, name='newBench'),
    # ex: /polls/start/go/
    url(r'^(?P<exp_id>[0-9]+)/goBench/$', views.goBench, name='goBench'),

    ]
