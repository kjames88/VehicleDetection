import numpy as np
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

svc = LinearSVC()
X_scaler = StandardScaler()
hog_orientations = 16

def color_hist(img):
    # as in lecture; [0] output is bin counts
    color0_hist = np.histogram(img[:,:,0], bins=32, range=(0,256))
    color1_hist = np.histogram(img[:,:,1], bins=32, range=(0,256))
    color2_hist = np.histogram(img[:,:,2], bins=32, range=(0,256))
    features = np.concatenate((color0_hist[0], color1_hist[0], color2_hist[0]))
    return features



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
    print(len(cars))

    images = glob.glob('non-vehicles/GTI/*.png')
    for image in images:
        not_cars.append(image)
    print(len(not_cars))

    # get hog features and scale them, following lecture code example
    car_features = []
    not_car_features = []
    for image in cars:
        #img = cv2.cvtColor(mpimg.imread(image), cv2.COLOR_RGB2HLS)
        img = mpimg.imread(image)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_norm = normalize(img[:,:,0])
        features = []
        if False:
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(hog(img[:,:,channel], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                                   transform_sqrt=True, visualise=False, feature_vector=True))
            features.append(np.ravel(hog_features))
        else:
            hog_features = hog(img_gray, orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                               transform_sqrt=True, visualise=False, feature_vector=True)
            features.append(hog_features)
        features.append(color_hist(img))
        car_features.append(np.concatenate(features))
    for image in not_cars:
        #img = cv2.cvtColor(mpimg.imread(image), cv2.COLOR_RGB2HLS)
        img = mpimg.imread(image)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_norm = normalize(img[:,:,0])
        features = []
        if False:
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(hog(img[:,:,channel], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                                   transform_sqrt=True, visualise=False, feature_vector=True))
            features.append(np.ravel(hog_features))
        else:
            hog_features = hog(img_gray, orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                               transform_sqrt=True, visualise=False, feature_vector=True)
            features.append(hog_features)
        features.append(color_hist(img))
        not_car_features.append(np.concatenate(features))
    X = np.vstack((car_features, not_car_features)).astype(np.float64)
    print('X shape {}'.format(X.shape))
    X_scaler.fit(X)
    scaled_X = X_scaler.transform(X)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_car_features))))
    print(scaled_X.shape)
    print(y.shape)
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
    print('x range 0 - {}, y range {} - {}'.format(x_dim, y_start, y_end))
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



def classify(hog_features, search_windows, scale=(64,64)):
    ymin = search_windows[0][0][1]
    hits = []
    for window in search_windows:
        # for small windows there is rounding error in selection of hog features
        y = np.int(7 * (window[0][1] - ymin) / scale[1])
        x = np.int(7 * (window[0][0]) / scale[0])
        # get a (7,7,2,2,9) array for each window (scaled to 64x64)
        hog_f = np.ravel(hog_features[y:y+7, x:x+7])
        color_f = color_hist(img[window[0][1]:window[1][1], window[0][0]:window[1][0]])
        features = np.concatenate((hog_f, color_f))
        scaled_X = X_scaler.transform(features.reshape(1,-1))
        pred = svc.predict(scaled_X)
        if pred == 1:
            hits.append(window)
    return hits 

def my_hog(img, search_windows):
    hits = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_norm = normalize(img[:,:,0])
    print('range is {} to {}'.format(np.min(img_norm), np.max(img_norm)))
    #img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    for window in search_windows:
        w = cv2.resize(img_gray[window[0][1]:window[1][1],window[0][0]:window[1][0]], (64,64))
        hog_features = hog(w,
                           orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                           transform_sqrt=True, visualise=False, feature_vector=False)
        color_f = color_hist(img[window[0][1]:window[1][1],window[0][0]:window[1][0]])
        features = np.concatenate((np.ravel(hog_features), color_f))
        scaled_X = X_scaler.transform(np.ravel(features).reshape(1,-1))
        pred = svc.predict(scaled_X)
        if pred == 1:
            hits.append(window)

    return hits


# compute hog for the frame then section it with sliding window for classification
def whole_hog(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if False:
        features = []
        for channel in range(img.shape[2]):
            hog_features = hog(img[:,:,channel], orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                               transform_sqrt=True, visualise=False, feature_vector=False)
            features.append(hog_features)
        return features
    else:
        hog_features = hog(img_gray, orientations=hog_orientations, pixels_per_cell=(8,8), cells_per_block=(2,2),
                           transform_sqrt=True, visualise=False, feature_vector=False)
        return hog_features


img = mpimg.imread('test_images/test1.jpg')
def search_frame(img):
    # start at middle of frame vertically and use an increasing window size approaching the bottom of frame
    search_windows = []
    hot_windows = []

    # small scale
    #   rescale a strip of the frame to 64x64 equivalent block size, get hog features, and subsample
    y = [400, 496]
    search_windows = sliding_window(img, y_range=y, window_size=(32,32), overlap=0.5)
    hot_windows = my_hog(img, search_windows)
    #scaled_img = resize_window(img[400:432, :], (16,16))
    #hog_features = whole_hog(scaled_img)
    #hot_windows = classify(hog_features, search_windows, (16,16))
    for w in hot_windows:
        cv2.rectangle(img, w[0], w[1], (0,0,255), 2)

    # medium scale
    y = [400, 528]
    s = (64,64)
    search_windows = sliding_window(img, y_range=y, window_size=(64,64), overlap=0.75)
    hot_windows = my_hog(img, search_windows)
    #scaled_img = resize_window(img[y[0]:y[1], :], (64,64))
    #hog_features = whole_hog(scaled_img)
    #hot_windows = classify(hog_features, search_windows, (64,64))
    for w in hot_windows:
        cv2.rectangle(img, w[0], w[1], (0,0,255), 2)

    # large scale
    #search_windows = sliding_window(img, y_range=[400, 596], window_size=(128,128), overlap=0.75)
    y = [400, 576]
    s = (128,128)
    search_windows = sliding_window(img, y_range=y, window_size=s, overlap=0.75)
    hot_windows = my_hog(img, search_windows)
    #scaled_img = resize_window(img[y[0]:y[1], :], s)
    #hog_features = whole_hog(scaled_img)
    #hot_windows = classify(hog_features, search_windows, s)
    for w in hot_windows:
        cv2.rectangle(img, w[0], w[1], (0,0,255), 2)
    
    y = [500, 675]
    search_windows = sliding_window(img, y_range=y, window_size=(256,256), overlap=0.75)
    hot_windows = my_hog(img, search_windows)
    for w in hot_windows:
        cv2.rectangle(img, w[0], w[1], (0,0,255), 2)

    plt.imshow(img)
    plt.show()


train()
search_frame(img)

