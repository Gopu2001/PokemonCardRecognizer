from django.db import models

# Create your models here.

# Upload a single image
class Image(models.Model):
	title = models.CharField(max_length=200)
	image = models.ImageField(upload_to='images/')

	def __str__(self):
		return self.title

# Upload six images at once (Pokemon Cards)
class SixImage(models.Model):
	title = models.CharField(max_length=200)
	image1 = models.ImageField(upload_to='images/', blank=False)
	image2 = models.ImageField(upload_to='images/', blank=True)
	image3 = models.ImageField(upload_to='images/', blank=True)
	image4 = models.ImageField(upload_to='images/', blank=True)
	image5 = models.ImageField(upload_to='images/', blank=True)
	image6 = models.ImageField(upload_to='images/', blank=True)

	def __str__(self):
		return self.title

# Create a new user
class User(models.Model):
	fname = models.CharField(max_length=100, name="First Name")
	lname = models.CharField(max_length=100, name="Last Name")
	uname = models.CharField(max_length=100, name="Username")
	email = models.EmailField(max_length=200, name="Email")
	passw = models.CharField(max_length=200, name="Password")

	def __str__(self):
		text  = "USER REGISTRATION FORM"
		text += "Username    Email        First Name    Last Name     "
		text += f"{self.uname.ljust(11)} {self.email.ljust(12)} {self.fname.ljust(13)} {self.lname.ljust(13)}"
		return text

class UserLogin(models.Model):
	uname = models.CharField(max_length=100, name="Username")
	passw = models.CharField(max_length=200, name="Password")

	def __str__(self):
		return f"USER LOGIN: {self.uname}"

class UserLogout(models.Model):
	def __str__(self):
		return "USER LOGOUT"
