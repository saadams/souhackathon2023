import os
import platform
#print(os.name)
user_plt = platform.system()
#user_plt = 'Windows'
print(platform.system())

if user_plt.lower() == "darwin":
    print('Mac User')
elif user_plt.lower() == "windows":
    print('windows user')
elif user_plt.lower() == 'linux':
    print("linux user")



#mac = Darwin
#windows = Windows
#linux = linux