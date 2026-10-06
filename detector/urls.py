from django.urls import path
from . import views

# Application URL routes
urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('detect/', views.detect_image, name='detect'),
    path('comparison/', views.model_comparison, name='comparison'),
    path('history/', views.prediction_history, name='history'),
    path('about/', views.about, name='about'),

    path('signup/', views.signup_view, name='signup'),
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('logout/', views.logout_view, name='logout'),
]