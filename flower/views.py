from django.shortcuts import redirect, render

from .forms import ImageForm

# Create your views here.
from .models import Image
from .recognition.recognize import main

def showall(request):
    images = Image.objects.all().order_by("-pk")
    context = {"images": images}
    return render(request, "flower/showall.html", context)

def upload(request):
    if request.method == "POST":
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            img = Image.objects.get(pk=Image.objects.count())
            output = main(img.picture)
            img.first_name = output[0][0]
            img.first_value = output[0][1]
            img.second_name = output[1][0]
            img.second_value = output[1][1]
            img.third_name = output[2][0]
            img.third_value = output[2][1]
            img.fourth_name = output[3][0]
            img.fourth_value = output[3][1]
            img.save()
            return redirect("flower:result")
    else:
        form = ImageForm()
    context = {"form": form}
    return render(request, "flower/upload.html", context)

def result(request):
    images = Image.objects.all().order_by("-pk")
    context = {"images": images[1:], "now_image": images[0]}
    return render(request, "flower/result.html", context)
