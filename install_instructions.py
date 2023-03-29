import os
import platform
#print(os.name)
user_plt = platform.system()
#user_plt = 'Windows'
#sprint(platform.system())

if user_plt.lower() == "darwin":
    print('Mac User Deteced!')
    print('Mac users follow the instructions found here for installing tensorflow: https://developer.apple.com/metal/tensorflow-plugin/')
    print("Mac users can run this following command in the project dir. 'pip install -r requirements_mac.txt' ")

elif user_plt.lower() == "windows":
    print('Windows user Deteced!')
    print("Please run the following command in the project dir.  'pip install -r requirements.txt' ")
    print("Windows users can find additional instructions to install tensorflow provided here: https://www.tensorflow.org/install/pip#cpu ")

elif user_plt.lower() == 'linux':
    print("linux user")
    print('Follow the instructions here: https://www.tensorflow.org/install/pip#linux')



#mac = Darwin
#windows = Windows
#linux = linux