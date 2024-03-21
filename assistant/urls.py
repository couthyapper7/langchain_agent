from django.urls import path
from .views import ask,create_token,refresh_token

urlpatterns = [
    path('ask/', ask),
    path('login/', create_token),
    path('refresh/', refresh_token),
]