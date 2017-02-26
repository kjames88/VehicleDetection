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

    # may be more useful to histogram only S

    # as in lecture; [0] output is bin counts
    color0_hist = np.histogram(img[:,:,0], bins=32)
    color1_hist = np.histogram(img[:,:,1], bins=32)
    color2_hist = np.histogram(img[:,:,2], bins=32)
    features = np.concatenate((color0_hist[0], color1_hist[0], color2_hist[0]))
    
    # size of this feature is small compared to HOG and spatial
    #   to be useful it needs to be increased in weight
    features = color2_hist[0]
    #for copies in range(0,29):
    #    features = np.concatenate((features, color2_hist[0]))
    return features



def spatial(img):
    # size of this feature is the correct order of magnitude vs HOG
    scaled = cv2.resize(img, (32,32))
    return scaled.ravel()



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
    #features.append(color_hist(ycrcb))
    features.append(spatial(ycrcb))
    return np.concatenate(features)



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
def sliding_window(img, y_range=[None, None], window_size=(64,64), overlap=0.5):
    x_dim = img.shape[1]
    if y_range[0] == None:
        y_start = 0
    else:
        y_start = y_range[0]
    if y_range[1] == None:
        y_end = img.shape[0]
    else:
        y_end = y_range[1]
    #print('x range 0 - {}, y range {} - {}'.format(x_dim, y_start, y_end))
    step_y = np.int(window_size[1] * (1.0 - overlap))
    hold_y = np.int(window_size[1] * overlap)  # overlap section of window to hold back in computing steps
    y_steps = np.int((y_end - y_start - hold_y) / step_y)
    step_x = np.int(window_size[0] * (1.0 - overlap))
    hold_x = np.int(window_size[0] * overlap)  # overlap section of window to hold back in computing steps
    x_steps = np.int((x_dim - hold_x) / step_x)
    windows = []
    for y in range(0,y_steps):
        for x in range(0,x_steps):
            start_y = y_start + (y * step_y)
            end_y = start_y + window_size[1]
            start_x = (x * step_x)
            end_x = start_x + window_size[0]
            windows.append(((start_x, start_y), (end_x, end_y)))
    return windows


# resize image to equivalent (64,64) scale
def resize_window(img, scale=(64,64)):
    scaled_x = np.int((img.shape[1] / scale[0]) * 64.0)
    scaled_y = np.int((img.shape[0] / scale[1]) * 64.0)
    img_scaled = cv2.resize(img, (scaled_x, scaled_y))
    print('initial size {}, scaled size {}'.format(img.shape, img_scaled.shape))
    return img_scaled



def classify(img, hog_features, search_windows, start=(0,0), scale=(64,64)):
    ymin = start[1]
    hits = []
    for window in search_windows:
        w_rgb = img[window[0][1]:window[1][1],window[0][0]:window[1][0]]
        # for small windows there is rounding error in selection of hog features
        y = np.int(7 * (window[0][1] - ymin) / scale[1])
        x = np.int(7 * (window[0][0]) / scale[0])
        # get a (7,7,2,2,9) array for each window (scaled to 64x64)
        hog_f = np.ravel(hog_features[y:y+7, x:x+7])
        color_f = color_hist(w_rgb)
        spatial_f = spatial(w_rgb)
        features = np.concatenate((hog_f, color_f, spatial_f))
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
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if hog_all_channels:
        features = []
        for channel in range(ycrcb.shape[2]):
            hog_features = hog(ycrcb[:,:,channel], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                               transform_sqrt=True, visualise=False, feature_vector=False)
            features.append(hog_features)
        return features
    else:
        hog_features = hog(ycrcb, orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                           transform_sqrt=True, visualise=False, feature_vector=False)
        return hog_features


def search_frame(img):
    # start at middle of frame vertically and use an increasing window size approaching the bottom of frame
    search_windows = []
    hot_windows = []

    # small scale
    y = [400, 528]
    s = (64,64)
    search_windows = sliding_window(img, y_range=y, window_size=s, overlap=0.75)
    hot_windows = (my_hog(img, search_windows))
    #scaled_img = resize_window(img[y[0]:y[1], :], s)
    #hog_features = whole_hog(scaled_img)
    #hot_windows.extend(classify(img, hog_features, search_windows, scale=s, start=(0,y[0])))

    y = [400, 560]
    s = (96,96)
    search_windows = sliding_window(img, y_range=y, window_size=s, overlap=0.75)
    hot_windows.extend(my_hog(img, search_windows))

    # medium scale
    y = [500, 656]
    s = (128,128)
    search_windows = sliding_window(img, y_range=y, window_size=s, overlap=0.75)
    hot_windows.extend(my_hog(img, search_windows))
    #scaled_img = resize_window(img[y[0]:y[1], :], s)
    #hog_features = whole_hog(scaled_img)
    #hot_windows.extend(classify(img, hog_features, search_windows, scale=s, start=(0,y[0])))

    # large scale
    #y = [400, 656]
    #s = (192,192)
    #search_windows = sliding_window(img, y_range=y, window_size=s, overlap=0.75)
    #hot_windows.extend(my_hog(img, search_windows))
   
    #for w in hot_windows:
    #    cv2.rectangle(img, w[0], w[1], (0,0,255), 2)
#    plt.imshow(img)
#    plt.show()

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


def temporal_filter(heat, labels):
    global heat_q
    global heat_2q

    # mask heat map using 2 stages of delay
    hot = np.copy(heat)
    if heat_q != None and heat_2q != None:
        hot[((heat_q == 0) | (heat_2q == 0))] = 0

    if True:
        # keep current heat map iff label contains masked hot pixels
        hot2 = np.copy(heat)
        for car_number in range(1, labels[1]+1):
            s = np.sum((labels[0] == car_number) & (hot > 0))
            if s == 0:  # no overlap between hot pixels and this label
                hot2[(labels[0] == car_number)] = 0  # erase this 'car' from heat map
        hot_labels = label(hot2)
    else:
        hot_labels = label(hot)
    if heat_q != None:
        heat_2q = np.copy(heat_q)
    heat_q = np.copy(heat)
    #hot_labels = label(hot)
    print(hot_labels[1], 'hot cars found')
    return hot_labels



def integrate_heat(heat):
    global heat_q
    global delay
    if delay == None:
        delay = np.zeros_like(heat)
    else:
        delay[heat > 0] += 1
        delay[heat == 0] = 0
    if heat_q == None:
        heat_q = np.copy(heat)
    else:
        heat_q[delay >= 5] -= 1
        heat_q[heat > 0] += 1
    delay += 1
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



def process_image(img):
    hot_windows = search_frame(img)
    heat, labels = heat_map(img, hot_windows)
    hot_labels = integrate_heat(heat)


    # DETAILED SEARCH


    #draw_labeled_bboxes(img, labels)
    draw_labeled_bboxes(img, hot_labels, (255,0,0))
    img = draw_heat(img, heat)
    return img


#img = mpimg.imread('test_images/test1.jpg')
#w = np.float32(img / 255)

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

def main():
    train()
    project_output = 'project_post.mp4'
    clip1 = VideoFileClip('project_video.mp4').subclip(14,30)
    #clip1 = VideoFileClip('test_video.mp4')
    project_clip = clip1.fl_image(process_image)  # color images required
    project_clip.write_videofile(project_output, audio=False)



if __name__ == "__main__":
    # execute only if run as a script
    main()



#plt.imshow(img)
#plt.show()

