# CycleGan: image-to-image transformation for unpaired dataset

This project provides a complete tool to use CycleGan, a method for image-to-image transformation without a need for paired dataset. 
Here I primarily use it to accomplish image-to-image transformation, specifically, between google satellite images and pre-1920 Japanese Arts.

To generate the following images, I webscraped about 2500 Japanese art pieces from the Library of Congress (https://www.loc.gov/collections/?fa=subject:art+%26+architecture) and the satellite maps from google.  
10 ~ 40 epoch seems to generate the best images, with Cycleloss of 10 ~ 12.5

Google Satellite | Pre-1920 Japanese Art Style
----------- | ------------
![google satellite](https://github.com/Davidnh8/CycleGan/blob/master/data/Japanese/Samples_Images/original_uofc.jpg_20_1_11.5.jpg) | ![Japanese Art](https://github.com/Davidnh8/CycleGan/blob/master/data/Japanese/Samples_Images/transformed_uofc.jpg_20_1_12.5.jpg)
![google satellite](https://github.com/Davidnh8/CycleGan/blob/master/data/Japanese/Samples_Images/original_GhostLake.jpg_10_1_12.5.jpg) | ![Japanese Art](https://github.com/Davidnh8/CycleGan/blob/master/data/Japanese/Samples_Images/transformed_GhostLake.jpg_45_1_11.5.jpg)
![google satellite](https://github.com/Davidnh8/CycleGan/blob/master/data/Japanese/Samples_Images/original_K1.jpg_10_1_11.5.jpg) | ![Japanese Art](https://github.com/Davidnh8/CycleGan/blob/master/data/Japanese/Samples_Images/transformed_K1.jpg_20_1_12.5.jpg)
