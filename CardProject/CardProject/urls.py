"""CardProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from cardAnalysis import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.redirect_to_profile),
	path('user/<str:username>/upload/', views.image_upload_view),
	path('new.user/', views.create_new_user),
	path('success/', views.success),
	path('failure/', views.failure),
	path('login/', views.user_login),
	path('logout/', views.user_logout),
	path('user/<str:username>/', views.view_profile, name='profile'),
	path('my/', views.my_profile),
]

if settings.DEBUG:
	urlpatterns += static(settings.MEDIA_URL,
						  document_root=settings.MEDIA_ROOT)
