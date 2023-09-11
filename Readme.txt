Overview:

The project extracts the feature descriptors from the Caltech101 dataset images, stores them into a local instance of MongoDB database,
and then consumes those database entries to find "k" images similar to the user enterred image.


System Requirements:

1. Python 3.4.11+
2. MongoDB Community Server v7.0.1
3. MongoDB Compass v1.39.4
4. Python libraries as stated below
5. Stable internet connection to download the dataset


Python libraries

1. torch
2. torchvision
3. pymongo
4. numpy
5. matplotlib
6. math
7. scipy


Steps:

Task 1:

1. Run the command "python ExtractFeatureDescriptors.py" from within the project directory. (or "python3 ExtractFeatureDescriptors.py" depedning upon the python alias in your system.)
2. Enter Image ID of choice when prompted.
3. Enter the desired feature model option as prompted.

The output will be displayed on the terminal.


Task 2:

(Make sure you have installed MongoDB requirements as stated above.)
1. Change the host specification, database name and collection name on line 9, 15, 18 respectively if necessary.
2. Run the command "python StoreToMongo.py" from within the project directory. (or "python3 StoreToMongo.py" depedning upon the python alias in your system.)

The data will be stored in the MongoDB collection.


Task 3:

(Make sure you have installed MongoDB requirements as stated above.)
1. Change the host specification, database name and collection name on line 10, 22, 25 respectively if necessary.
2. Run the command "python FindKImages.py" from within the project directory. (or "python3 FindKImages.py" depending upon the python alias in your system.)
3. Enter the image ID of choice when prompted.
4. Enter "k" when prompted.

The matplotlib grid with the similar images will be displayed.