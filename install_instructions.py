import os
import platform
#print(os.name)
user_plt = platform.system()
#user_plt = 'Windows'
#sprint(platform.system())

if user_plt.lower() == "darwin":
    print('Mac User Deteced!')
    print('Mac users follow the instructions found here for installing tensorflow: https://developer.apple.com/metal/tensorflow-plugin/')
    print("Mac users can run this following command. 'pip install -r requirements_mac.txt' ")

elif user_plt.lower() == "windows":
    print('Windows user Deteced!')
    print("Please run the following command in the project dir.  'pip install -r requirements.txt' ")
    print("Or follow the instructions provided here")

elif user_plt.lower() == 'linux':
    print("linux user")
    print('Follow the instructions here: ')



#mac = Darwin
#windows = Windows
#linux = linux