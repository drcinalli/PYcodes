from django.contrib.auth.models import User, Group
from rest_framework import serializers
from service.models import Player, Play


class PlayerSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Player
        fields = ('id', 'url', 'username', 'email', 'password', 'name', 'schooling', 'gender', 'age')


class GroupSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Group
        fields = ('url', 'name')

class PlaySerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Play
        fields = ('id', 'level', 'chromosomeOne', 'chromosomeOneIndex', 'chromosomeTwo', 'chromosomeTwoIndex', 'answer', 'answer_time', 'points', 'play_player', 'play_experiment')