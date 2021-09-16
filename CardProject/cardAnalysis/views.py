from django.shortcuts import render
from .forms import SixImageForm, CreateNewUserForm, LoginForm, LogoutForm
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.core.exceptions import PermissionDenied
from django.http import HttpResponseRedirect, HttpResponse
from pprint import pprint
import os

# Create your views here.

def image_upload_view(request, username):
	"""Process images uploaded by users"""
	if request.user.username != username:
		request.session['failure message'] = f'Access Denied! You ({request.user.username}) are not authorized to upload images for this user ({username})!'
		raise PermissionDenied(request.session['failure message'])

	if request.method == "POST":
		form = SixImageForm(request.POST, request.FILES, username=request.user.username)
		if form.is_valid():
			form.save()
			form_data = form.instance
			#handle_uploaded_image(username,
			#					 form_data.image1)
			#form_data.image1.save()
			image1, image2, image3, image4, image5, image6, title = (form_data.image1, form_data.image2,
																	 form_data.image3, form_data.image4,
																	 form_data.image5, form_data.image6,
																	 form_data.title)
			return render(request, 'upload.html', {'form' : form,
												  'title' : title,
												  'image1' : image1,
												  'image2' : image2,
												  'image3' : image3,
												  'image4' : image4,
												  'image5' : image5,
												  'image6' : image6})


	else:
		form = SixImageForm(username=request.user.username)

	return render(request, 'upload.html', {'form' : form})

#def handle_uploaded_image(username, f):
#	print(username, type(f))
#	print(os.getcwd())
#	if username in os.listdir('media/images/'):
#		os.mkdir(f'media/images/{username}')
#	with open(f'images/{username}/1.jpg', 'w+') as file:
#		for chunk in f.chunks():
#			file.write(chunk)

def create_new_user(request):
	"""Process User registration requests"""

	if request.method == "POST":
		form = CreateNewUserForm(request.POST)
		if form.is_valid():
			form_data = form.instance
			#pprint(dir(form_data))
			fname, lname, uname, email, passw = (form_data.__dict__['First Name'],
												 form_data.__dict__['Last Name'],
												 form_data.__dict__['Username'],
												 form_data.__dict__['Email'],
												 form_data.__dict__['Password'])
			user = User.objects.create_user(username=uname, email=email, password=passw)
			user.fname = fname
			user.lname = lname
			user.save()

			return HttpResponseRedirect('/success/')

	else:
		form = CreateNewUserForm()
		#pprint(dir(form))

	return render(request, 'new.user.html', {'form' : form})

def success(request):
	"""Display success message"""
	username = request.user.username
	print(f"User: {username}")

	return HttpResponse(f"Success! Logged in user is now \"{username}\"")

def failure(request):
	"""Display failure message"""

	return HttpResponse(f"Failure! {request.session['failure message'] if request.session['failure message'] is not None else ''}")

def user_login(request):
	"""Log a user in"""
	if request.method == "POST":
		form = LoginForm(request.POST)
		if form.is_valid():
			form_data = form.instance
			uname, passw = (form_data.__dict__['Username'],
							form_data.__dict__['Password'])

			user = authenticate(request, username=uname, password=passw)
			if user is not None:
				login(request, user)
				print("Successfully logged in!")
				return HttpResponseRedirect('/success/')
			else:
				print("Incorrect username or password... Please try again")
				return HttpResponseRedirect('/failure/')
		else:
			print("The form is not valid?!?!")
			return HttpResponseRedirect('/failure/')

	else:
		form = LoginForm()
		return render(request, 'login.html', {'form' : form})

	print("Something went awry... Please submit a request to the Admin")
	return HttpResponseRedirect('/failure/')

def user_logout(request):
	"""Log a user out"""
	if request.method == "POST":
		form = LogoutForm(request.POST)
		if form.is_valid():
			logout(request)
			return HttpResponseRedirect('/success/')

	else:
		form = LogoutForm()
		return render(request, 'logout.html', {'form' : form})

	return HttpResponseRedirect('/failure/')

def view_profile(request, username):
	"""View a public profile"""
	if not request.user.is_authenticated:
		return HttpResponseRedirect('/login/')
	return HttpResponse(f"You are {request.user} viewing {username}'s profile! You are {' ' if request.user.is_authenticated else 'not '}logged in!")

def my_profile(request):
	return HttpResponseRedirect(f"/user/{request.user}/")

def redirect_to_profile(reequest):
	return HttpResponseRedirect("/my/")
