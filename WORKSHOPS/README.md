# Workshops
ESB2025 and GRAZ2023 contain the codes, data, and slides for each workshop. 

# General Instructions
1. If you have never used Google Colab before, you will need to right-click on `BrainCANN.ipynb` > Open with > Connect more apps > Colaboratory to install Google Colab onto your account.
2. Download `CANNsBRAINdata.xlsx` and `BrainCANN.ipynb` and upload them to your Google Drive account (for GRAZ2023 download the `input` folder).
3. Carefully read the instructions in the Jupyter Notebook for installing packages.
3. Change the `path` variable to direct to where you put these files, as noted in the code comments.
4. Run the code blocks sequentially.

# Input: brain data, same as in the BRAIN folder
Uniaxial tension/compression and simple shear data for the cortext, corona radiata, corpus callosum, basal ganglia.

# Debugging Common Issues
1. Make sure you are using the correct matplotlib and tensorflow versions by running the second code block where you can check the versions. After you run `!pip install ...` you first need to restart the runtime (runtime > restart runtime) and then import the packages. 
3. Make sure all the code blocks have been checked; missing one would prevent the code from running.
4. In the event that the code is still not running, please add `!pip install pandas==1.5.3` and `!pip install numpy==1.23.5` and restart the runtime. Google colab updates packages on a regular basis which may cause this version of the code to break. The package versions listed below are compatible with this code. 

# Common Questions
Q. Do I need to install anything? \
A. No, Colab is a cloud-based computing platform that is free to use with a Google account. Python and many common packages come pre-installed.

# Compatible Package Versions
`matplotlib==3.2.2` \
`tensorflow==2.12.0` \
`numpy==1.23.5` \
`pandas==1.5.3` \
`sklearn==1.2.2` \
`json==2.0.9` \
`python==3.10.12`
