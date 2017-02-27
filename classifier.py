import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.ndimage.measurements import label
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip


def color_hist(img):
    # as in lecture; [0] output is bin counts
    color0_hist = np.histogram(img[:,:,0], bins=32)
    color1_hist = np.histogram(img[:,:,1], bins=32)
    color2_hist = np.histogram(img[:,:,2], bins=32)
    features = np.concatenate((color0_hist[0], color1_hist[0], color2_hist[0]))
    return features



def spatial(img):
    # size of this feature is the correct order of magnitude vs HOG
    scaled = cv2.resize(img, (32,32))
    return scaled.ravel()



def get_features_nohog(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    features = []
    features.append(color_hist(ycrcb))
    features.append(spatial(ycrcb))
    return np.concatenate(features)



def get_features(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    features = []
    if hog_all_channels:
        hog_features = []
        for channel in range(ycrcb.shape[2]):
            hog_features.append(hog(ycrcb[:,:,channel], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                               transform_sqrt=True, visualise=False, feature_vector=True))
        features.append(np.ravel(hog_features))
    elif hog_gray:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hog_features = hog(img_gray, orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                           transform_sqrt=True, visualise=False, feature_vector=True)
        features.append(hog_features)
    else:
        hog_features = hog(ycrcb[:,:,hog_channel], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                           transform_sqrt=True, visualise=False, feature_vector=True)
        features.append(hog_features)
    features.append(color_hist(ycrcb))
    features.append(spatial(ycrcb))
    return np.concatenate(features)



# NOTE:  make sure the input is scaled to 0-1 range
def get_prediction(img, window):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    sample = cv2.resize(ycrcb[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64,64))
    features = get_features(sample)
    scaled_X = X_scaler.transform(np.ravel(features).reshape(1,-1))
    pred = svc.predict(scaled_X)
    return pred
   
 

def train():
    cars = []
    not_cars = []
    images = glob.glob('vehicles/GTI_Far/*.png')
    for image in images:
        cars.append(image)
    images = glob.glob('vehicles/GTI_MiddleClose/*.png')
    for image in images:
        cars.append(image)
    images = glob.glob('vehicles/GTI_Left/*.png')
    for image in images:
        cars.append(image)
    images = glob.glob('vehicles/GTI_Right/*.png')
    for image in images:
        cars.append(image)
    images = glob.glob('vehicles/KITTI_extracted/*.png')
    for image in images:
        cars.append(image)
    print('cars samples', len(cars))
#
    images = glob.glob('non-vehicles/GTI/*.png')
    for image in images:
        not_cars.append(image)
    images = glob.glob('non-vehicles/Extras/*.png')
    for image in images:
        not_cars.append(image)
    print('not car samples', len(not_cars))

    # get hog features and scale them, following lecture code example
    car_features = []
    not_car_features = []
    Mright = np.float32([[1,0,8], [0,1,8]])
    Mleft = np.float32([[1,0,-8], [0,1,-8]])
    for image in cars:
        img = mpimg.imread(image)
        rows,cols,channels = img.shape
        car_features.append(get_features(img))
        #img_shift_right = cv2.warpAffine(img, Mright, (cols,rows))
        #img_shift_left = cv2.warpAffine(img, Mleft, (cols,rows))
        #car_features.append(get_features(img_shift_right))
        #car_features.append(get_features(img_shift_left))

    for image in not_cars:
        img = mpimg.imread(image)
        rows,cols,channels = img.shape
        not_car_features.append(get_features(img))
        #img_shift_right = cv2.warpAffine(img, Mright, (cols,rows))
        #img_shift_left = cv2.warpAffine(img, Mleft, (cols,rows))
        #not_car_features.append(get_features(img_shift_right))
        #not_car_features.append(get_features(img_shift_left))

    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    X_scaler.fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)
    svc.fit(X_train, y_train)
    pred = svc.predict(X_test)
    score = accuracy_score(pred, y_test)
    print('accuracy {}'.format(score))



# generate a list of sliding windows in image; for searching classification matches
# similar to lecture function
def sliding_window(img, y_range=[None, None], x_range=[None, None], window_size=(64,64), step=16):
    x_dim = img.shape[1]
    if y_range[0] == None:
        y_start = 0
    else:
        y_start = max(y_range[0], 0)
    if y_range[1] == None:
        y_end = img.shape[0]
    else:
        y_end = min(y_range[1], img.shape[0])
    if x_range[0] == None:
        x_start = 0
    else:
        x_start = max(x_range[0], 0)
        
    if x_range[1] == None:
        x_end = img.shape[1]
    else:
        x_end = min(x_range[1], img.shape[1])
    #print('x range 0 - {}, y range {} - {}'.format(x_dim, y_start, y_end))
    step_y = step
    hold_y = window_size[1] - step  # overlap section of window to hold back in computing steps
    y_steps = np.int((y_end - y_start - hold_y) / step_y)
    step_x = step
    hold_x = window_size[0] - step  # overlap section of window to hold back in computing steps
    x_steps = np.int((x_end - x_start - hold_x) / step_x)
    windows = []
    for y in range(0,y_steps):
        for x in range(0,x_steps):
            start_y = y_start + (y * step_y)
            end_y = start_y + window_size[1]
            start_x = x_start + (x * step_x)
            end_x = start_x + window_size[0]
            windows.append(((start_x, start_y), (end_x, end_y)))
    return windows


# resize image to equivalent (64,64) scale
def resize_window(img, scale=(64,64)):
    scaled_x = np.int((img.shape[1] / scale[0]) * 64.0)
    scaled_y = np.int((img.shape[0] / scale[1]) * 64.0)
    img_scaled = cv2.resize(img, (scaled_x, scaled_y))
    return img_scaled



# resize image to equivalent (64,64) block-size scale
def resize_image(img, scale=1.0):
    scaled_x = np.int(img.shape[1]/scale)
    scaled_y = np.int(img.shape[0]/scale)
    img_scaled = cv2.resize(img, (scaled_x, scaled_y))
    return img_scaled



# Whole-frame HOG classifier
#   uses the sub-sampled results from whole_hog() along with input-resolution spatial and color features
def classify(img, hog_features, search_windows, scale=1.0):
    tfm_img = np.float32(img/255)  # video image has range 0-255 vs 0-1 for png training images
    hits = []
    for window in search_windows:
        w_rgb = tfm_img[window[0][1]:window[1][1],window[0][0]:window[1][0]]
        scaled_y = np.int(window[0][1] / scale)  # original sliding window position in scaled image
        scaled_x = np.int(window[0][0] / scale)

        # for small windows there is rounding error in selection of hog features
        y = np.int(8 * (scaled_y / 64))  # scaling changes the block size to 64x64 effective and there are 7x7 feature arrays per block
        x = np.int(8 * (scaled_x / 64))
        # get a (7,7,2,2,9) array for each window (scaled to 64x64)
        hog_f = []
        if hog_all_channels:
            hf = []
            for channel in range(0,tfm_img.shape[2]):
                hf.append(np.ravel(hog_features[channel][y:y+7, x:x+7]))
            hog_f = np.ravel(hf)
        else:
            hog_f = np.ravel(hog_features[y:y+7, x:x+7])
        features = np.concatenate((hog_f, get_features_nohog(w_rgb)))
        scaled_X = X_scaler.transform(features.reshape(1,-1))
        pred = svc.predict(scaled_X)
        if pred == 1:
            hits.append(window)
    return hits 



def my_hog(img, search_windows):
    hits = []
    for window in search_windows:
        w = cv2.resize(img[window[0][1]:window[1][1],window[0][0]:window[1][0]], (64,64))
        w = np.float32(w/255)  # video image has range 0-255 vs 0-1 for png training images
        features = get_features(w)
        scaled_X = X_scaler.transform(np.ravel(features).reshape(1,-1))
        pred = svc.predict(scaled_X)
        if pred == 1:
            hits.append(window)

    return hits


# compute hog for the frame then section it with sliding window for classification
def whole_hog(img):
    tfm_img = np.float32(img/255)  # video image has range 0-255 vs 0-1 for png training images
    ycrcb = cv2.cvtColor(tfm_img, cv2.COLOR_RGB2YCrCb)
    if hog_all_channels:
        hog_features = []
        for channel in range(ycrcb.shape[2]):
            hf = hog(ycrcb[:,:,channel], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                     transform_sqrt=True, visualise=False, feature_vector=False)
            hog_features.append(hf)
        return hog_features
    else:
        hog_features = hog(ycrcb[:,:,hog_channel], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                           transform_sqrt=True, visualise=False, feature_vector=False)
        return hog_features



# This is an alternative whole-frame HOG method that generates its own sliding window using a stepsize
#   Adapted from the discussion video due to trouble with whole_hog+classify
#   Needs scale input to generate 96x96 and 128x128 results
def search_frame_v2(img):
    # instead of using sliding window just run 64x64 native scale with whole-image HOG
    scaled_img = np.float32(img/255)  # video image has range 0-255 vs 0-1 for png training images
    ycrcb = cv2.cvtColor(scaled_img, cv2.COLOR_RGB2YCrCb)
    hog_ch0 = hog(ycrcb[:,:,0], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                  transform_sqrt=True, visualise=False, feature_vector=False)
    hog_ch1 = hog(ycrcb[:,:,1], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                  transform_sqrt=True, visualise=False, feature_vector=False)
    hog_ch2 = hog(ycrcb[:,:,2], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                  transform_sqrt=True, visualise=False, feature_vector=False)
    cells_per_step = 2
    ycells = (ycrcb.shape[0] // 8)
    xcells = (ycrcb.shape[1] // 8)
    steps_per_block = np.int(8 / cells_per_step)
    ysteps = (ycells // cells_per_step) - steps_per_block
    xsteps = (xcells // cells_per_step) - steps_per_block
    ystart = 25
    windows = []
    for y in range(ystart,ysteps):
        for x in range(xsteps):
            features = []
            hog_y = (y * cells_per_step)
            hog_x = (x * cells_per_step)
            hog_features_ch0 = hog_ch0[hog_y:hog_y+7, hog_x:hog_x+7].ravel()
            hog_features_ch1 = hog_ch1[hog_y:hog_y+7, hog_x:hog_x+7].ravel()
            hog_features_ch2 = hog_ch2[hog_y:hog_y+7, hog_x:hog_x+7].ravel()
            features.append(hog_features_ch0)
            features.append(hog_features_ch1)
            features.append(hog_features_ch2)
            #hog_features = np.hstack((hog_features_ch0, hog_features_ch1, hog_features_ch2))
            #print('hstack shape {}'.format(hog_features.shape))
            start_y = y * (8 * cells_per_step)
            end_y = start_y + 64
            start_x = x * (8 * cells_per_step)
            end_x = start_x + 64
            #features.append(hog_features)
            feature_img = ycrcb[start_y:end_y, start_x:end_x]
            features.append(color_hist(feature_img))
            features.append(spatial(feature_img))
            features = np.concatenate(features)
            scaled_X = X_scaler.transform(features.reshape(1,-1))
            pred = svc.predict(scaled_X)
            if pred == 1:
                windows.append(((start_x, start_y), (end_x, end_y)))
    return windows


def search_frame(img):
    # start at middle of frame vertically and use an increasing window size approaching the bottom of frame
    search_windows = []
    hot_windows = []
    count = 0

    # small scale
    y = [400, 528]
    s = (64,64)
    search_windows = sliding_window(img, y_range=y, window_size=s, step=16)
    count += len(search_windows)
    #hot_windows.extend(my_hog(img, search_windows))
    scaled_img = resize_image(img, 1.0)
    hog_features = whole_hog(scaled_img)
    hot_windows.extend(classify(img, hog_features, search_windows, scale=1.0))

    y = [400, 560]
    s = (96,96)
    search_windows = sliding_window(img, y_range=y, window_size=s, step=16)
    count += len(search_windows)
    #hot_windows.extend(my_hog(img, search_windows))
    scaled_img = resize_image(img, 1.5)
    hog_features = whole_hog(scaled_img)
    hot_windows.extend(classify(img, hog_features, search_windows, scale=1.5))

    # medium scale
    y = [500, 656]
    s = (128,128)
    search_windows = sliding_window(img, y_range=y, window_size=s, step=32)
    count += len(search_windows)
    #hot_windows.extend(my_hog(img, search_windows))
    scaled_img = resize_image(img, 2.0)
    hog_features = whole_hog(scaled_img)
    hot_windows.extend(classify(img, hog_features, search_windows, scale=2.0))

    print('searched {} windows'.format(count))
    return hot_windows



# create a thresholded heat map to remove false positives and duplicates
#    - from lecture
def heat_map(img, windows):
    heat = np.zeros_like(img[:,:,0])
    for w in windows:
        heat[w[0][1]:w[1][1], w[0][0]:w[1][0]] += 1
    heat[heat < 3] = 0  # threshold to remove areas with too few hits
    labels = label(heat)
    #print(labels[1], 'cars found')
    return heat, labels


#def temporal_filter(heat, labels):
#    global heat_q
#    global heat_2q
#
#    # mask heat map using 2 stages of delay
#    hot = np.copy(heat)
#    if heat_q != None and heat_2q != None:
#        hot[((heat_q == 0) | (heat_2q == 0))] = 0
#
#    if True:
#        # keep current heat map iff label contains masked hot pixels
#        hot2 = np.copy(heat)
#        for car_number in range(1, labels[1]+1):
#            s = np.sum((labels[0] == car_number) & (hot > 0))
#            if s == 0:  # no overlap between hot pixels and this label
#                hot2[(labels[0] == car_number)] = 0  # erase this 'car' from heat map
#        hot_labels = label(hot2)
#    else:
#        hot_labels = label(hot)
#    if heat_q != None:
#        heat_2q = np.copy(heat_q)
#    heat_q = np.copy(heat)
#    #hot_labels = label(hot)
#    print(hot_labels[1], 'hot cars found')
#    return hot_labels



def integrate_heat(heat):
    global heat_q
    global delay
    if delay == None:
        delay = np.zeros_like(heat)
    else:
        delay[heat == 0] += 1
        delay[heat > 0] = 0
    if heat_q == None:
        heat_q = np.copy(heat)
    else:
        heat_q[(heat_q > 0) & (delay >= 5)] -= 1
        heat_q[heat > 0] += 1
    hot_labels = label(heat_q)
    print(hot_labels[1], 'hot cars found')
    return hot_labels



# function copied from lecture (35)
def draw_labeled_bboxes(img, labels, color=(0,0,255)):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 5)
    # Return the image
    return img



def draw_heat(img, heat):
    heat_color = np.zeros_like(img)
    color = np.zeros_like(heat)
    color[heat > 0] = 255
    heat_color[:,:,2] = color
    overlay = cv2.addWeighted(img, 1.0, heat_color, 0.5, 0)
    return overlay



# using part of draw_labeled_bboxes to recover the boxes for refinement
def identify_cars(img, labels):
    # Iterate through all detected cars
    hot_windows = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        pos = (np.min(nonzeroy), np.min(nonzerox))  # (y, x)
        size = (np.max(nonzeroy) - np.min(nonzeroy), np.max(nonzerox) - np.min(nonzerox))
        print('car {} at y={} x={} size {}x{}'.format(car_number, pos[0], pos[1], size[0], size[1]))
        #if size[0] > 64 or size[1] > 64:
        #    if size[0] >= 96 or size[1] >= 96:
        #        search_size = (128, 128)
        #    else:
        #        search_size = (96,96)
        #else:
        #    search_size = (64,64)
        search_size = (32,32)
        windows = sliding_window(img, (pos[0]-32, pos[0]+size[0]+32), (pos[1]-32, pos[1]+size[1]+32), search_size, step=16)
        scaled_img = np.float32(img/255)
        for w in windows:
            pred = get_prediction(scaled_img, w)
            hot_windows.append(w)
    heat, labels = heat_map(img, hot_windows)
    return heat, labels



def process_image(img):
    hot_windows = search_frame(img)
    heat, labels = heat_map(img, hot_windows)
    hot_labels = integrate_heat(heat)
    draw_labeled_bboxes(img, hot_labels)
    img = draw_heat(img, heat)
    heat, hot_labels = identify_cars(img, hot_labels)  # DETAILED SEARCH
    draw_labeled_bboxes(img, hot_labels, (255,0,0))
    return img


svc = LinearSVC(class_weight='balanced')
#svc = SVC(kernel='poly', degree=2, class_weight='balanced')
X_scaler = StandardScaler()
hog_all_channels = True
hog_gray = False
hog_channel = 0
hog_orientations = 9
heat_q = None
heat_2q = None
delay = None

#img = mpimg.imread('test_images/test1.jpg')
#search_frame_v2(img)

def main():
    train()
    project_output = 'project_post.mp4'
    #clip1 = VideoFileClip('project_video.mp4')
    clip1 = VideoFileClip('test_video.mp4')
    project_clip = clip1.fl_image(process_image)  # color images required
    project_clip.write_videofile(project_output, audio=False)



if __name__ == "__main__":
    # execute only if run as a script
    main()



#plt.imshow(img)
#plt.show()

