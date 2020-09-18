#from django.conf.urls import patterns, include, url
from django.contrib import admin

#urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'web.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    #url(r'^admin/', include(admin.site.urls)),
#)

from django.conf.urls import url, include
from rest_framework import routers
from service import views
from django.contrib.auth.decorators import login_required

from administration import views as v

router = routers.DefaultRouter()
router.register(r'players', views.PlayerViewSet)
#router.register(r'play', views.PlayViewSet)
#router.register(r'groups', views.GroupViewSet)

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browseable API.
urlpatterns = [
	url(r'^$', include('administration.urls')),
	url(r'^experiments/$', v.experiments),
    url(r'^interfaces/$', v.interfaces),

    url(r'^service/login/(?P<resource_id>\d+)[/]?$', (views.LoginUserView.as_view()), name='my_rest_view'),
    url(r'^service/login[/]?$', (views.LoginUserView.as_view()), name='my_rest_view'),

    #nao precisa estar no servico
    #url(r'^service/start_experiment/(?P<resource_id>\d+)[/]?$', views.StartExperimentView.as_view(), name='start_experiment'),
    #url(r'^service/start_experiment[/]?$', views.StartExperimentView.as_view(), name='start_experiment'),

    url(r'^service/get_area/(?P<resource_id>\d+)[/]?$', views.GetAreaView.as_view(), name='get_area'),
    url(r'^service/get_area[/]?$', views.GetAreaView.as_view(), name='get_area'),

    url(r'^service/ready_to_play/(?P<resource_id>\d+)[/]?$', views.ReadyToPlayView.as_view(), name='ready_to_play'),
    url(r'^service/ready_to_play[/]?$', views.ReadyToPlayView.as_view(), name='ready_to_play'),

    #FAKE PRA TESTE DO ROGERIO
    url(r'^service/ready_to_play_fake/(?P<resource_id>\d+)[/]?$', views.ReadyToPlayFakeView.as_view(), name='ready_to_play_fake'),
    url(r'^service/ready_to_play_fake[/]?$', views.ReadyToPlayFakeView.as_view(), name='ready_to_play_fake'),

    url(r'^service/user_registration/(?P<resource_id>\d+)[/]?$', views.UserRegistrationView.as_view(), name='register_user'),
    url(r'^service/user_registration[/]?$', views.UserRegistrationView.as_view(), name='register_user'),

    url(r'^service/get_plays/(?P<username>.+)[/]?$', views.GetPlaysView.as_view(), name='get_plays'),
    url(r'^service/get_plays[/]?$', views.GetPlaysView.as_view(), name='get_plays'),

    url(r'^service/get_rank/(?P<username>.+)[/]?$', views.GetRankView.as_view(), name='get_rank'),
    url(r'^service/get_rank[/]?$', views.GetRankView.as_view(), name='get_rank'),

    url(r'^service/start_experiment/(?P<username>.+)[/]?$', views.StartExperimentView.as_view(), name='start_experiment'),
    url(r'^service/start_experiment[/]?$', views.StartExperimentView.as_view(), name='start_experiment'),

    url(r'^service/send_result/(?P<username>.+)[/]?$', views.SendResultView.as_view(), name='send_result'),
    url(r'^service/send_result[/]?$', views.SendResultView.as_view(), name='send_result'),

    #url(r'^service/login/', views.LoginUserView),
    url(r'^service/', include(router.urls)),
    url(r'^api-auth/', include('rest_framework.urls', namespace='rest_framework')),

    url(r'^polls/', include('polls.urls', namespace="polls")),
    url(r'^admin/', include(admin.site.urls))
]
