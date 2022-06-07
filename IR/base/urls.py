from django.urls import path
from . import views

urlpatterns = [
    path("", views.init, name="init"),
    path("\home", views.home, name="home"),
    path("\Get1SuggResult", views.GetSuggResultfrom1, name="Sugg1"),
    path("\Get2SuggResult", views.GetSuggResultfrom2, name="Sugg2"),
    path("\Advanced_1", views.advanced_word_embadding_1, name="advancedModel"),
    path("\Advanced_2", views.advanced_word_embadding_2, name="advancedMode2"),
    path("\wordEmbadding1", views.get_first_Results, name="normal1"),
    path("\wordEmbadding2", views.get_second_Results, name="normal2"),
]
