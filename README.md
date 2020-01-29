# Person-and-helmet-counter
Counts the number of person and helmet in images with motorcycle

This code combines 2 codes:
  1. Person counter
    git@github.com:nancyshaji/PeopleCounter.git
  2. Helmet counter
    https://github.com/BlcaKHat/yolov3-Helmet-Detection.git
    
 The file module2.py has the main code. While the other two python files have the other two counters for your reference.
 
 Add the input images to the images folder.
 
 Things to download:
    yolov3.weights,
    yolov3-obj_2400.weights
    
  Download the files from Darknet site and https://drive.google.com/file/d/16yH9M_ovw0cJG4gVKuXTkz_cwYxJtwAk/view respectively
  
  After that run the module2.py file. 
  
  Weaknesses are:
      1. Takes caps as helmets
      2. Both the model are not 100 percent accurate.
      
 This has been developed as part of our projects(Helmet detection), module 2.
