from django.urls import path
from .views import *
from rest_framework.authtoken.views import obtain_auth_token
urlpatterns = [
    path('ask/', ask),
    path('register/', UserCreate.as_view(), name='user-register'),
]