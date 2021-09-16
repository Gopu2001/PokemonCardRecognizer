from django import forms
from .models import Image, SixImage, User, UserLogin, UserLogout


class ImageForm(forms.ModelForm):
	"""Form for the image model"""

	class Meta:
		model = Image
		fields = ('title', 'image')

class SixImageForm(forms.ModelForm):
	"""Form for the image model"""

	def __init__(self, *args, **kwargs):
		self.username = kwargs.pop('username')
		super().__init__(*args, **kwargs)

	def save(self, *args, **kwargs):
		# for each of the uploaded files, save them to the username's folder
		super().save(*args, **kwargs)
		for filename, file in self.files.items():
			print(filename, file)
			print(type(file.photo))

	class Meta:
		model = SixImage
		fields = ('title', 'image1', 'image2', 'image3', 'image4', 'image5', 'image6')

class CreateNewUserForm(forms.ModelForm):
	"""Form for creating a new account"""

	class Meta:
		model = User
		fields = ('First Name', 'Last Name', 'Username', 'Email', 'Password')

class LoginForm(forms.ModelForm):
	"""Form for Logging in a User"""

	class Meta:
		model = UserLogin
		fields = ('Username', 'Password')

class LogoutForm(forms.ModelForm):
	"""Form for logging out a User"""

	class Meta:
		model = UserLogout
		fields = ()
