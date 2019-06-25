from __future__ import print_function, division, unicode_literals
import glob
import os
import PIL.Image
import numpy as np
import pickle
from viz import plot_seg_label_distributions

__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


#calculate_class_weights
#BDD100K
label_colormap = {
    "unlabeled"         :(  0,  0,  0),
    "dynamic"           :(111, 74,  0),
    "ego vehicle"       :(  0,  0,  0),
    "ground"            :( 81,  0, 81),
    "static"            :(  0,  0,  0),
    "parking"           :(250,170,160),
    "rail track"        :(230,150,140),
    "road"              :(128, 64,128),
    "sidewalk"          :(244, 35,232),
    "bridge"            :(150,100,100),
    "building"          :( 70, 70, 70),
    "fence"             :(190,153,153),
    "garage"            :(180,100,180),
    "guard rail"        :(180,165,180),
    "tunnel"            :(150,120, 90),
    "wall"              :(102,102,156),
    "banner"            :(250,170,100),
    "billboard"         :(220,220,250),
    "lane divider"      :(255, 165, 0),
    "parking sign"      :(220, 20, 60),
    "pole"              :(153,153,153),
    "polegroup"         :(153,153,153),
    "street light"      :(220,220,100),
    "traffic cone"      :(255, 70,  0),
    "traffic device"    :(220,220,220),
    "traffic light"     :(250,170, 30),
    "traffic sign"      :(220,220,  0),
    "traffic sign frame":(250,170,250),
    "terrain"           :(152,251,152),
    "vegetation"        :(107,142, 35),
    "sky"               :( 70,130,180),
    "person"            :(220, 20, 60),
    "rider"             :(255,  0,  0),
    "bicycle"           :(119, 11, 32),
    "bus"               :(  0, 60,100),
    "car"               :(  0,  0,142),
    "caravan"           :(  0,  0, 90),
    "motorcycle"        :(  0,  0,230),
    "trailer"           :(  0,  0,110),
    "train"             :(  0, 80,100),
    "truck"             :(  0,  0, 70),
}
id2label = [
    'unlabeled'         ,
    'dynamic'           ,
    'ego vehicle'       ,
    'ground'            ,
    'static'            ,
    'parking'           ,
    'rail track'        ,
    'road'              ,
    'sidewalk'          ,
    'bridge'            ,
    'building'          ,
    'fence'             ,
    'garage'            ,
    'guard rail'        ,
    'tunnel'            ,
    'wall'              ,
    'banner'            ,
    'billboard'         ,
    'lane divider'      ,
    'parking sign'      ,
    'pole'              ,
    'polegroup'         ,
    'street light'      ,
    'traffic cone'      ,
    'traffic device'    ,
    'traffic light'     ,
    'traffic sign'      ,
    'traffic sign frame',
    'terrain'           ,
    'vegetation'        ,
    'sky'               ,
    'person'            ,
    'rider'             ,
    'bicycle'           ,
    'bus'               ,
    'car'               ,
    'caravan'           ,
    'motorcycle'        ,
    'trailer'           ,
    'train'             ,
    'truck'             ,
]
'''label_colormap = {
    "Banner":(250,170,100),
    "Billboard":(220,220,250),
    "LaneDivider":(255,165,0),
    "ParkingSign":(220,20,60),
    "Pole":(153,153,153),
    "PoleGroup":(153,153,153),
    "StreetLight":(220,220,100),
    "TrafficCone":(255,70,0),
    "TrafficDevice":(220,220,220),
    "TrafficLight":(250,170,30),
    "TrafficSign":(220,220,0),
    "SignFrame":(250,170,250),
    "Person":(220,20,60),
    "Rider":(255,0,0),
    "Bicycle":(119,11,32),
    "Bus":(0,60,100),
    "Car":(0,0,142),
    "Caravan":(0,0,90),
    "Motorcycle":(0,0,230),
    "Trailer":(0,0,110),
    "Train":(0,80,100),
    "Truck":(0,0,70),
}

id2label = [
    'Banner',
    'Billboard',
    'LaneDivider',
    'ParkingSign',
    'Pole',
    'PoleGroup',
    'StreetLight',
    'TrafficCone',
    'TrafficDevice',
    'TrafficLight',
    'TrafficSign',
    'SignFrame',
    'Person',
    'Rider',
    'Bicycle',
    'Bus',
    'Car',
    'Caravan',
    'Motorcycle',
    'Trailer',
    'Train',
    'Truck',
]'''
'''labels = [
    #       name                     id    trainId   category catId      hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  1 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ego vehicle'          ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ground'               ,  3 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'parking'              ,  5 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           ,  6 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'bridge'               ,  9 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'building'             , 10 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'fence'                , 11 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'garage'               , 12 ,      255 , 'construction'    , 2       , False        , True         , (180,100,180) ),
    Label(  'guard rail'           , 13 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'tunnel'               , 14 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'wall'                 , 15 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'banner'               , 16 ,      255 , 'object'          , 3       , False        , True         , (250,170,100) ),
    Label(  'billboard'            , 17 ,      255 , 'object'          , 3       , False        , True         , (220,220,250) ),
    Label(  'lane divider'         , 18 ,      255 , 'object'          , 3       , False        , True         , (255, 165, 0) ),
    Label(  'parking sign'         , 19 ,      255 , 'object'          , 3       , False        , False        , (220, 20, 60) ),
    Label(  'pole'                 , 20 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 21 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'street light'         , 22 ,      255 , 'object'          , 3       , False        , True         , (220,220,100) ),
    Label(  'traffic cone'         , 23 ,      255 , 'object'          , 3       , False        , True         , (255, 70,  0) ),
    Label(  'traffic device'       , 24 ,      255 , 'object'          , 3       , False        , True         , (220,220,220) ),
    Label(  'traffic light'        , 25 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 26 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'traffic sign frame'   , 27 ,      255 , 'object'          , 3       , False        , True         , (250,170,250) ),
    Label(  'terrain'              , 28 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'vegetation'           , 29 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'sky'                  , 30 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 31 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 32 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'bus'                  , 34 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'car'                  , 35 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'caravan'              , 36 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'motorcycle'           , 37 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'trailer'              , 38 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 39 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'truck'                , 40 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
]
#camvid
label_colormap = {
    "Animal": (64, 128, 64),
    "Archway": (192, 0, 128),
    "Bicyclist": (0, 128, 192),
    "Bridge": (0, 128, 64),
    "Building": (128, 0, 0),
    "Car": (64, 0, 128),
    "CartLuggagePram": (64, 0, 192),
    "Child": (192, 128, 64),
    "Column_Pole": (192, 192, 128),
    "Fence": (64, 64, 128),
    "LaneMkgsDriv": (128, 0, 192),
    "LaneMkgsNonDriv": (192, 0, 64),
    "Misc_Text": (128, 128, 64),
    "MotorcycleScooter": (192, 0, 192),
    "OtherMoving": (128, 64, 64),
    "ParkingBlock": (64, 192, 128),
    "Pedestrian": (64, 64, 0),
    "Road": (128, 64, 128),
    "RoadShoulder": (128, 128, 192),
    "Sidewalk": (0, 0, 192),
    "SignSymbol": (192, 128, 128),
    "Sky": (128, 128, 128),
    "SUVPickupTruck": (64, 128, 192),
    "TrafficCone": (0, 0, 64),
    "TrafficLight": (0, 64, 64),
    "Train": (192, 64, 128),
    "Tree": (128, 128, 0),
    "Truck_Bus": (192, 128, 192),
    "Tunnel": (64, 0, 64),
    "VegetationMisc": (192, 192, 0),
    "Void": (0, 0, 0),
    "Wall": (64, 192, 0),
}


id2label = [
    'Void',
    'Sky',
    'Pedestrian',
    'Child',
    'Animal',
    'Tree',
    'VegetationMisc',
    'CartLuggagePram',
    'Bicyclist',
    'MotorcycleScooter',
    'Car',
    'SUVPickupTruck',
    'Truck_Bus',
    'Train',
    'OtherMoving',
    'Road',
    'RoadShoulder',
    'Sidewalk',
    'LaneMkgsDriv',
    'LaneMkgsNonDriv',
    'Bridge',
    'Tunnel',
    'Archway',
    'ParkingBlock',
    'TrafficLight',
    'SignSymbol',
    'Column_Pole',
    'Fence',
    'TrafficCone',
    'Misc_Text',
    'Wall',
    'Building',
]'''

label2id = {label:id for id, label in enumerate(id2label)}
idcolormap = [label_colormap[label] for label in id2label]

# Check nothing stupid happened with mappings
assert set(label_colormap) == set(id2label) == set(label2id.keys()), "Something is wrong with the id label maps"

# ==============================================================================
#                                                                 MAYBE_MAKE_DIR
# ==============================================================================
def maybe_make_dir(path):
    """ Checks if a directory path exists on the system, if it does not, then
        it creates that directory (and any parent directories needed to
        create that directory)
    """
    if not os.path.exists(path):
        os.makedirs(path)

# ==============================================================================
#                                                                     GET_PARDIR
# ==============================================================================
def get_pardir(file):
    """ Given a file path, it returns the parent directory of that file. """
    return os.path.dirname(file)


# ==============================================================================
#                                                              MAYBE_MAKE_PARDIR
# ==============================================================================
def maybe_make_pardir(file):
    """ Takes a path to a file, and creates the necessary directory structure
        on the system to ensure that the parent directory exists (if it does
        not already exist)
    """
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)


# ==============================================================================
#                                                                       FILE2STR
# ==============================================================================
def file2str(file):
    """ Takes a file path and returns the contents of that file as a string."""
    with open(file, "r") as textFile:
        return textFile.read()

# ==============================================================================
#                                                                       STR2FILE
# ==============================================================================
def str2file(s, file, mode="w"):
    """ Writes a string to a file"""
    # Ensure parent directory and necesary file structure exists
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)

    with open(file, mode=mode) as textFile:
        textFile.write(s)


# ==============================================================================
#                                                                     OBJ2PICKLE
# ==============================================================================
def obj2pickle(obj, file, protocol=2):
    """ Saves an object as a binary pickle file to the desired file path. """
    # Ensure parent directory and necesary file structure exists
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)

    with open(file, mode="wb") as fileObj:
        pickle.dump(obj, fileObj, protocol=protocol)


# ==============================================================================
#                                                                     PICKLE2OBJ
# ==============================================================================
def pickle2obj(file):
    """ Loads the contents of a pickle as a python object. """
    with open(file, mode = "rb") as fileObj:
        obj = pickle.load(fileObj)
    return obj


# ==============================================================================
#                                                              CREATE_FILE_LISTS
# ==============================================================================
def create_file_lists(inputs_dir, labels_dir):
    """ Given the paths to the directories containing the input and label
        images, it creates a list of the full filepaths for those images,
        with the same ordering, so the same index in each list represents
        the corresponding input/label pair.

        Returns 2-tuple of two lists: (input_files, label_files)
    """
    # Create (synchronized) lists of full file paths to input and label images
    '''label_files = glob.glob(os.path.join(labels_dir, "*.png"))
    file_ids = [os.path.basename(f).replace("_L.png", ".png") for f in label_files]'''

    label_files = glob.glob(os.path.join(labels_dir, "*.png"))
    #print(label_files)
    #file_ids = [os.path.basename(f).replace("_L.png", ".png") for f in label_files]
    file_ids = [os.path.basename(f).replace("_train_color.png", ".jpg") for f in label_files]   

    input_files = [os.path.join(inputs_dir, file_id) for file_id in file_ids]
    return input_files, label_files


# ==============================================================================
#                                                               CREATE_DATA_DICT
# ==============================================================================
def create_data_dict(datadir, X_train_subdir="train_inputs", Y_train_subdir="train_labels"):
    data = {}
    data["X_train"], data["Y_train"] = create_file_lists(
        inputs_dir=os.path.join(datadir, X_train_subdir),
        labels_dir=os.path.join(datadir, Y_train_subdir))
    return data


# ==============================================================================
#                                                              PIXELS_WITH_VALUE
# ==============================================================================
def pixels_with_value(img, val):
    """ Given an image as a numpy array, and a value representing the
        pixel values, eg [128,255,190] in an RGB image, then it returns
        a 2D boolean array with a True for every pixel position that has
        that value.
    """
    return np.all(img==np.array(val), axis=2)


# ==============================================================================
#                                                                   RGB2SEGLABEL
# ==============================================================================
def rgb2seglabel(img, colormap, channels_axis=False):
    """ Given an RGB image stored as a numpy array, and a colormap that
        maps from label id to an RGB color for that label, it returns a
        new numpy array with color chanel size of 1 where the pixel
        intensity values represent the class label id.

    Args:
        img:            (np array)
        colormap:       (list) list of pixel values for each class label id
        channels_axis:  (bool)(default=False) Should it return an array with a
                        third (color channels) axis of size 1?
    """
    height, width, _ = img.shape
    if channels_axis:
        label = np.zeros([height, width,1], dtype=np.uint8)
    else:
        label = np.zeros([height, width], dtype=np.uint8)
    '''print(label.shape)
    arr = np.all(img==np.array(idcolormap[16]))
    print(arr)
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()'''
    for id in range(len(colormap)):
        label[np.all(img==np.array(idcolormap[id]), axis=2)] = id
    return label


# ==============================================================================
#                                                       LOAD_IMAGE_AND_SEGLABELS
# ==============================================================================
def load_image_and_seglabels(input_files, label_files, colormap, shape=(32,32), n_channels=3, label_chanel_axis=False):
    """ Given a list of input image file paths and corresponding segmentation
        label image files (with different RGB values representing different
        classes), and a colormap list, it:

        - loads up the images
        - resizes them to a desired shape
        - converts segmentation labels to single color channel image with
          integer value of pixel representing the class id.

    Args:
        input_files:        (list of str) file paths for input images
        label_files:        (list of str) file paths for label images
        colormap:           (list or None) A list where each index represents the
                            color value for the corresponding class id.
                            Eg: for RGB labels, to map class_0 to black and
                            class_1 to red:
                                [(0,0,0), (255,0,0)]
                            Set to None if images are already encoded as
                            greyscale where the integer value represents the
                            class id.
        shape:              (2-tuple of ints) (width,height) to reshape images
        n_channels:         (int) Number of chanels for input images
        label_chanel_axis:  (bool)(default=False) Use color chanel axis for
                            array of label images?
    """
    # Dummy proofing
    assert n_channels in {1,3}, "Incorrect value for n_channels. Must be 1 or 3. Got {}".format(n_channels)

    # Image dimensions
    width, height = shape
    n_samples = len(label_files)

    # Initialize input and label batch arrays
    X = np.zeros([n_samples, height, width, n_channels], dtype=np.uint8)
    if label_chanel_axis:
        Y = np.zeros([n_samples, height, width, 1], dtype=np.uint8)
    else:
        Y = np.zeros([n_samples, height, width], dtype=np.uint8)
    print(n_samples)
    for i in range(n_samples):
        # Get filenames of input and label
        img_file = input_files[i]
        label_file = label_files[i]

        # Resize input and label images
        img = PIL.Image.open(img_file).resize(shape, resample=PIL.Image.CUBIC)
        
        label_img   = PIL.Image.open(label_file).resize(shape, resample=PIL.Image.NEAREST).convert('RGB')

        '''img = PIL.Image.open(img_file).resize(shape, resample=PIL.Image.CUBIC)
        
        label_img   = PIL.Image.open(label_file).resize(shape, resample=PIL.Image.NEAREST).convert('RGB')'''
        #label_img = img2[:,:,:3]

        '''import matplotlib.pyplot as plt
        plt.imshow(label_img)
        plt.show()'''
        # Convert back to numpy arrays
        img = np.asarray(img, dtype=np.uint8)
        label_img = np.asarray(label_img, dtype=np.uint8)

        # Convert label image from RGB to single value int class labels
        if colormap is not None:
            label_img = rgb2seglabel(label_img, colormap=colormap, channels_axis=label_chanel_axis)

        # Add processed images to batch arrays
        X[i] = img
        Y[i] = label_img
        print(int(i/n_samples*100),"%",end='\r')
    return X, Y


# ==============================================================================
#                                                                   PREPARE_DATA
# ==============================================================================
def prepare_data(data_file, valid_from_train=False, n_valid=1024, max_data=None, verbose=True):
    data = pickle2obj(data_file)

    # Create validation from train data
    if valid_from_train:
        data["X_valid"] = data["X_train"][:n_valid]
        data["Y_valid"] = data["Y_train"][:n_valid]
        data["X_train"] = data["X_train"][n_valid:]
        data["Y_train"] = data["Y_train"][n_valid:]

    if max_data:
        data["X_train"] = data["X_train"][:max_data]
        data["Y_train"] = data["Y_train"][:max_data]

    # Visualization data
    n_viz = 25
    data["X_train_viz"] = data["X_train"][:25]
    data["Y_train_viz"] = data["Y_train"][:25]

    data["id2label"] = id2label
    data["label2id"] = label2id
    data["colormap"] = idcolormap

    if verbose:
        # Print information about data
        print("DATA SHAPES")
        print("- X_valid: ", (data["X_valid"]).shape)
        print("- Y_valid: ", (data["Y_valid"]).shape)
        print("- X_train: ", (data["X_train"]).shape)
        print("- Y_train: ", (data["Y_train"]).shape)
        if "X_test" in data:
            print("- X_test: ", (data["X_test"]).shape)
            print("- Y_test: ", (data["Y_test"]).shape)

    return data


# ==============================================================================
#                                                        CALCULATE_CLASS_WEIGHTS
# ==============================================================================
def calculate_class_weights(Y, n_classes, method="paszke", c=1.02):
    """ Given the training data labels Calculates the class weights.

    Args:
        Y:      (numpy array) The training labels as class id integers.
                The shape does not matter, as long as each element represents
                a class id (ie, NOT one-hot-vectors).
        n_classes: (int) Number of possible classes.
        method: (str) The type of class weighting to use.

                - "paszke" = use the method from from Paszke et al 2016
                            `1/ln(c + class_probability)`
                - "eigen"  = use the method from Eigen & Fergus 2014.
                             `median_freq/class_freq`
                             where `class_freq` is based only on images that
                             actually contain that class.
                - "eigen2" = Similar to `eigen`, except that class_freq is
                             based on the frequency of the class in the
                             entire dataset, not just images where it occurs.
                -"logeigen2" = takes the log of "eigen2" method, so that
                            incredibly rare classes do not completely overpower
                            other values.
        c:      (float) Coefficient to use, when using paszke method.

    Returns:
        weights:    (numpy array) Array of shape [n_classes] assigning a
                    weight value to each class.

    References:
        Eigen & Fergus 2014: https://arxiv.org/abs/1411.4734
        Paszke et al 2016: https://arxiv.org/abs/1606.02147
    """
    # CLASS PROBABILITIES - based on empirical observation of data
    ids, counts = np.unique(Y, return_counts=True)
    n_pixels = Y.size
    p_class = np.zeros(n_classes)
    p_class[ids] = counts/n_pixels

    # CLASS WEIGHTS
    if method == "paszke":
        weights = 1/np.log(c+p_class)
    elif method == "eigen":
        assert False, "TODO: Implement eigen method"
        # TODO: Implement eigen method
        # where class_freq is the number of pixels of class c divided by
        # the total number of pixels in images where c is actually present,
        # and median freq is the median of these frequencies.
    elif method in {"eigen2", "logeigen2"}:
        epsilon = 1e-8 # to prevent division by 0
        median = np.median(p_class)
        weights = median/(p_class+epsilon)
        if method == "logeigen2":
            weights = np.log(weights+1)
    else:
        assert False, "Incorrect choice for method"

    return weights


if __name__ == '__main__':
    # SETTINGS
    #data_dir = "/path/to/camvid"
    #data_dir = "/home/ronny/TEMP/camvid/"
    data_dir = "/home/artyom/Desktop/Datasets/tamp/"
    #data_dir = "/home/artyom/Desktop/Datasets/camvid/"
    pickle_file = "../pickle_jar/data_HD4.pickle"
    #pickle_file = "data_256.pickle"
    #Specify shape
    shape = [424,240]
    #shape = [256, 256]
    width, height = shape
    n_channels = 3
    label_chanel_axis=False # Create chanels axis for label images?

    print("CREATING DATA")
    print("- Getting list of files")
    file_data = create_data_dict(data_dir, X_train_subdir="train_inputs", Y_train_subdir="train_labels")
    n_samples = len(file_data["X_train"])

    est_size = n_samples*width*height*(3+1)/(1024*1000)
    print("- Estimated data size is {} MB (+ overhead)".format(est_size))

    print("- Loading image files and converting to arrays")
    data = {}
    data["X_train"], data["Y_train"] = load_image_and_seglabels(
        input_files=file_data["X_train"],
        label_files=file_data["Y_train"],
        colormap=idcolormap,
        shape=shape,
        n_channels=n_channels,
        label_chanel_axis=label_chanel_axis)

    # # Visualize the data
    from viz import viz_overlayed_segmentation_label
    from viz import batch2grid
    gx = batch2grid(data["X_train"], 5,5)
    gy = batch2grid(data["Y_train"], 5,5)
    g = viz_overlayed_segmentation_label(gx, gy, colormap=idcolormap, alpha=0.7, saveto=None)
    g.show()

    print("- Pickling the data to:", pickle_file)
    obj2pickle(data, pickle_file)
    print("- DONE!")
    #plot_seg_label_distributions(file_data["Y_train"], id2label, idcolormap)
