# Vehicle Detection and Tracking

## Project Description ##

This project implements a vehicle detection and tracking pipeline.  A LinearSVC classifier is first trained on a database of car images and non-car images.  This classifier is subsequently used with a sliding window analysis of video frames to detect the presence and location of cars.  Filtering is applied to reduce false positives and track the cars more smoothly.

## Methodology ##

The first step is to train a classifier.  For this project GTI and KITTI vehicle images are used, along with GTI and additional Udacity-sourced non-car images.  All images are 64x64 png files.  The following process is used for training.

1. Convert image to YCrCb color space
2. Compute Histogram of Oriented Gradients on each channel (Y, Cr, Cb)
3. Scale image to 32x32 and shape as 1-d (this is the 'spatial' feature)
4. Concatenate the HOG and spatial features into a 1-d feature vector
5. Fit a Standard Scaler to normalize columns of data individually
6. Create a label vector to associate car or not-car as a binary flag per feature vector
7. Split the samples with train-test split of 20% test data
8. Fit LinearSVC classifier to the training data
9. Predict the labels for the test data using the new classifier
10. Use test predictions to compute Accuracy Score of the new classifier

### HOG Features Examples ###

The following images show an example training car image, a non-car training image, and a video capture image of two cars.  Each image is followed by the HOG features visualization image for the Luma, Chroma-red and Chroma-blue channels of the image.  As shown below, the vehicle images have distinguishable features in the luma channel and one of the chroma channels.  The non-vehicle image is harder to recognize from the HOG features.  Hopefully the classifier also detects the lack of car-similarity.

#### Training Image of a Car ####

![car0074]
(output_images/car0074.png "Example training car image")

![car0074_hog0]
(output_images/car0074_hog0.png "Example training car HOG features Luma channel")

![car0074_hog1]
(output_images/car0074_hog1.png "Example training car HOG features Cr channel")

![car0074_hog2]
(output_images/car0074_hog2.png "Example training car HOG features Cb channel")

#### Training Image of a Non-Car ####

![non-car1415]
(output_images/non-car1415.png "Example training non-car image")

![non-car1415_hog0]
(output_images/non-car1415_hog0.png "Example training non-car HOG features Luma channel")

![non-car1415_hog1]
(output_images/non-car1415_hog1.png "Example training non-car HOG features Cr channel")

![non-car1415_hog2]
(output_images/non-car1415_hog2.png "Example training non-car HOG features Cb channel")

#### Freeway Test Image ####

![real cars]
(output_images/real_cars.png "Image from video stream (test1.jpg)")

![hog y]
(output_images/real_cars_hog0.png "HOG features Luma channel")

![hog_cr]
(output_images/real_cars_hog1.png "HOG features Cr channel")

![hog_cb]
(output_images/real_cars_hog2.png "HOG features Cb channel")

Using video input frames, the pipeline uses the following steps to locate and track vehicles.

1. Scale pixel value range from 8bit unsigned (0-255) to 32bit floating point (0-1.0)
2. Convert frame to YCrCb color space
3. Compute Histogram of Oriented Gradients on each channel (Y, Cr, Cb) of the frame
4. Search at three scales across the relevant region of the frame
5. For each search window scale to 64x64 and extract HOG features to form a feature vector
6. For each search window scale to 32x32 and add pixel values to feature vector
7. Use LinearSVC trained previously to predict whether the window contains a car
8. Create a heat map by aggregating pixels located as cars in the search windows
9. Threshold the heat map to reduce false positives
10. Label the heat map to identify candidate vehicles
11. Apply a temporal filter to smooth the vehicle detection across frames
12. Isolate the filtered vehicle bounding boxes and search a slightly larger region to determine a final bounding box
13. Draw the final bounding box on the outgoing frame

### Result Image ###

![output image]
(output_images/output1.png "Detected vehicles in video frame") 

## Discussion ##

### Classifier ###

The LinearSVC classifier is trained in the *train()* function.  Samples are read from the GTI, KITTI, and Extras folders.  HOG is computed using the default 8x8 cell size, 2x2 cells per block and 9 orientation bins.  Spatial features are 32x32 and no color histogram feature is included.  I experimented with 16, 24, 32 orientations and this gave some advantage up to 24 bins before I discovered the pixel value scaling problem.  After resolving that issue I remained with 9 orientations because performance was good.  I also experimented with 16x16 spatial resolution, with apparent success, but I switched back to 32x32 during testing and did not feel the need to change again.  The color histogram appeared to be harmful at some points and not useful at others.  The feature vectors are used to train the Standard Scaler instance, which is then used for scaling features of the video frames.

### Sliding Window Search ###

I use the full image horizontally but only y >= 384. Above that point are very small vehicles and scenery, so not worth the computational cost.  Search is at 64x64, 96x96, and 128x128 resolution, with 16 pixel steps for the two smaller sizes and 32 pixel steps for the largest.  Performance degraded with 96x96 stepped at 24-32 pixels.  I tried to maximize the phase-alignment potential with smaller steps while keeping the total search windows under 1k per recommendation.  The code is implemented in the *sliding_window()* and *search_frame()* functions.

### Whole-Frame HOG ###

I spent quite a bit of time on the whole-frame HOG optimization because of a math error in my code.  Before I found this, I followed the somewhat different approach in the instructor's video for 64x64 scale only.  Debugging my mistakes in that exercise revealed the bug in the earlier code.  My approach scales the image, computes HOG, and then computes the sliding window offset into the HOG features.  I use the standard resolution image to extract spatial and color features.  The observed performance improvement was less than stellar, possibly due to another change I made while watching the video; this was to scale and HOG the whole frame.  I had been cropping the frame and using additional offsets that I did not want to debug at the last minute.  This optimization is next on the list for future changes.  The code is implemented in the *whole_hog()* and *classify()* functions.  Note that *search_frame_v2()* implements the 64x64 instructor's video method which is currently not used.  The *my_hog()* function is used for computing window-by-window hog and is currently not used.

### Heat Map ###

The initial heat map (*heat_map()* function) follows the lectures.  The detected windows contribute +1 for each pixel they cover and the result is thresholded to reduce false positives.  Since the sliding window is expected to detect a vehicle at multiple offsets, the heat detected should not be too low.  A value of 1 or 2 suggests a false hit.

### Filtering ###

After the initial heat map, I used a temporal filter to smooth the detection and further reduce false positives.  Two consecutive frames will add a pixel, whereas five consecutive misses will drop the pixel.  In more detail, the filter increments (+1) when the input heat map is set and decrements when it is not set (-1), and resulting values less than 2 are dropped.  If a pixel is oscillating it can be kept to stabilize the detection, but after 5 consecutive misses it is dropped regardless of the count value.  This process is a bit tricky since adding too much heat and then reducing slowly can allow a bounding box to drag out and become much larger than the car before being trimmed much later.  There is room for improvement here, but the filter does a reasonably good job at this point.  The code is implemented in the *integrate_heat()* function.

### Boundary Box ###

The final bounding box drawn is typically larger than that resulting from the raw heat result described above.  I had observed that the bounding box sometimes understated the vehicle dimensions due to partial heat map retention.  I added a detailed search applying window-hog to a box somewhat larger than the raw heat result suggested.  The idea was to fill in parts of the vehicle that might have been missed by the sliding window.  The search is conducted at 64x64 resolution with the step size reduced from 16 to 8 pixels.  64x64 is a compromise that tends to extend the box beyond the car to a moderate extent, but typically avoids understating the car's dimensions.  The code for this is contained in the *identify_cars()* function.


