import os
import platform
#print(os.name)
user_plt = platform.system()
#user_plt = 'Windows'
print(platform.system())

if user_plt.lower() == "darwin":
    print('Mac User Deteced!')
    print('Mac users follow the instructions found here for installing tensorflow: ')
elif user_plt.lower() == "windows":
    print('Windows user Deteced!')
    print("Windows users can follow instructions found here: ")
    
elif user_plt.lower() == 'linux':
    print("linux user")



#mac = Darwin
#windows = Windows
#linux = linux