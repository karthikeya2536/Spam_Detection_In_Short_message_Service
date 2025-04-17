from django.contrib import messages
from django.shortcuts import render, redirect
from users.models import UserRegistrationModel


# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            # Store admin session data
            request.session['admin'] = 'admin'
            return redirect('AdminHome')
        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})


def AdminHome(request):
    # Check if admin is logged in
    if 'admin' not in request.session:
        messages.error(request, 'Please login to access this page')
        return redirect('AdminLogin')

    print("Rendering AdminHome")
    return render(request, 'admins/AdminHome.html')


def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request, 'admins/viewregisterusers.html', {'data': data})


def DeleteUser(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        UserRegistrationModel.objects.filter(id=id).delete()
        messages.success(request, 'User deleted successfully')
        return RegisterUsersView(request)


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request, 'admins/viewregisterusers.html', {'data': data})