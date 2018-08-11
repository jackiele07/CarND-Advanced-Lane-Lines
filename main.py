import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import pickle
import importlib
import os
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
cline = importlib.import_module('line_class')


# global variable
nwindow = 9
margin = 100
mpixel = 30
us_line_long = 30
us_line_wide = 3.7

left_line = cline.Line(nwindow, margin, mpixel, 0, us_line_long, us_line_wide)
right_line = cline.Line(nwindow, margin, mpixel, 0, us_line_long, us_line_wide)

# unit test for each processing step
# 1. cal_undistort
# 2. grd_color_thresh
# 3. extract_roi
# 4. process_image
# 5. perspective_transform
def sanity_check(img,fp,cmap=None,color=None):
    #img = cv2.imread('camera_cal/calibration1.jpg')
    output = fp(img)
    if ((len(output) ==  2) | (len(output) == 3)):
        out_image = np.copy(output[0])
    else:
        out_image = np.copy(output)
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.tight_layout()
    if(color==None):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        out_image = cv2.cvtColor(out_image,cv2.COLOR_BGR2RGB)
    ax1.imshow(img, cmap= cmap)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(out_image, cmap = cmap)
    ax2.set_title('Processed Image', fontsize=20)
    plt.show()

# unit test to save image for eyes-inspection
def sanity_check_save_image(img,fp,out_name):
    output = fp(img)
    if ((len(output) ==  2) | (len(output) == 3)):
        out_image = np.copy(output[0])
    else:
        out_image = np.copy(output)
    cv2.imwrite(out_name,out_image)

# Camera Calibration
def camera_calibration():
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    nx = 9
    ny = 6

    # get number of fail for tracking
    count = 0
    file_err = ""
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob('camera_cal/*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners2,ret)
            plt.imshow(img)
            plt.show()
        else:
            count += 1
            file_err += fname
            file_err += "\n"

    # Create dict to store objectpoints and imagepoints
    d = dict()
    d.update({'objpoints':objpoints})
    d.update({'imgpoints':imgpoints})

    with open('calibrate.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if (count > 0):
        with open('calibrate_erro_file.txt','w') as fp:
            fp.write(file_err)
    return

# Undistortion process
def cal_undistort(img):
    # Read in the saved objpoints and imgpoints
    dist_pickle = pickle.load( open( "calibrate.pickle", "rb" ) )
    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]

    if(len(img.shape) > 2):
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Calibrate camera and undistort
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

# Gradient & color thresholding
def grd_color_thresh(img, g_thresh=(170,255), s_thresh=(170, 255), sx_thresh=(20, 100)):
    # create local copy
    img = np.copy(img)

    # Get the r_channel to detect white color
    g_channel = img[:,:,2]
    # Threshold r_color channel
    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel >= g_thresh[0]) & (g_channel <= g_thresh[1])] = 1


    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Threshold s_color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Sobel x
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    sobely = cv2.Sobel(s_channel,cv2.CV_64F,0,1) # Take the derivative in y
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    magsobelxy = np.sqrt((sobelx**2) +(sobely**2))
    processed_sobelx = magsobelxy
    scaled_sobel = np.uint8(255*processed_sobelx/np.max(processed_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1


    # Stack each channel
    combined_binary = np.zeros_like(s_channel)
    combined_binary[(s_binary == 1) | (sxbinary == 1) | (g_binary == 1)] = 1
    return combined_binary

# extract region of interest
def extract_roi(img):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #hard coded for ROI - still could not apply dynamic ROI yet
    left_bot = [230, 690]
    left_top = [560, 400]
    right_top = [720, 400]
    right_bot = [1200, 690]

    vertices = np.array([[left_bot ,left_top, right_top, right_bot]], dtype=np.int32)
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, vertices

# process image
# smoothen image by Guassian Blur - apply thresholding & sobel - extract ROI
def process_image(img):
    cv2.GaussianBlur(img,(5, 5),0)
    processed_image = grd_color_thresh(img)
    extracted_image, vertices = extract_roi(processed_image)
    return extracted_image, vertices

# transform image to bird eyes view
def perspective_transform(img):
    non_transform_img, src = process_image(img)
    img_size = (non_transform_img.shape[1],non_transform_img.shape[0])

    # Source points should be changed to dynamic base on detection
    left_bot = [250, 678]
    left_top = [585, 456]
    right_top = [700, 456]
    right_bot = [1045, 678]
    src2 = np.array([[left_bot ,left_top, right_top, right_bot]], dtype=np.float32)

    dst = np.float32([[320, 720], [320, 0], [960, 0], [960, 720]])
    non_transform_img = cal_undistort(non_transform_img)
    M = cv2.getPerspectiveTransform(src2, dst)
    warped_img = cv2.warpPerspective(non_transform_img,M,img_size)
    # take M_revert to revert back from bird eye to image 2D
    Minv = cv2.getPerspectiveTransform(dst, src2)
    return warped_img, img, Minv

def test_perspective_transform():
    non_transform_img = cv2.imread("test_images/straight_lines1.jpg")
    img_size = (non_transform_img.shape[1],non_transform_img.shape[0])

    # Source points should be changed to dynamic base on detection
    left_bot = [250, 678]
    left_top = [585, 456]
    right_top = [700, 456]
    right_bot = [1045, 678]
    src2 = np.array([[left_bot ,left_top, right_top, right_bot]], dtype=np.float32)
    dst = np.float32([[320, 720], [320, 0], [960, 0], [960, 720]])
    non_transform_img = cal_undistort(non_transform_img)

    M = cv2.getPerspectiveTransform(src2, dst)
    warped_img = cv2.warpPerspective(non_transform_img,M,img_size)
    # take M_revert to revert back from bird eye to image 2D
    Minv = cv2.getPerspectiveTransform(dst, src2)

    cv2.line(non_transform_img, (250, 678), (585, 456), [255, 0, 0], 5)
    cv2.line(non_transform_img, (585, 456), (700, 456), [255, 0, 0], 5)
    cv2.line(non_transform_img, (1045, 678), (700, 456), [255, 0, 0], 5)
    cv2.line(non_transform_img, (250, 678), (1045, 678), [255, 0, 0], 5)

    non_transform_img = cv2.cvtColor(non_transform_img,cv2.COLOR_BGR2RGB)
    warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
    f, (ax1, ax2) = plt.subplots(1, 2)
    f.tight_layout()
    ax1.imshow(non_transform_img)
    ax1.set_title('Original Image with source point draw', fontsize=20)
    ax2.imshow(warped_img)
    ax2.set_title('Warped result', fontsize=20)
    plt.show()
    return warped_img

#init centroid of window in Line class
def init_centroid(wrapped_img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(wrapped_img[wrapped_img.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    left_centroid = np.argmax(histogram[:midpoint])
    right_centroid = np.argmax(histogram[midpoint:]) + midpoint

    return left_centroid, right_centroid

# detect lane base on bird eyes view
def detect_lane(img):
    wrapped_img,  undist, Minv = perspective_transform(img)
    left_centroid, right_centroid = init_centroid(wrapped_img)
    left_line.window.centroid = left_centroid
    right_line.window.centroid = right_centroid
    curvature = lane_mid = veh_pos = 0
    left_line.update_polycoeff(wrapped_img)
    right_line.update_polycoeff(wrapped_img)
    if(left_line.has_data == right_line.has_data == True):
        output = update_lane_detection(left_line, right_line, Minv, wrapped_img, undist)
        curvature = update_curvature()
        lane_mid = get_lane_mid()
        veh_pos = (img.shape[1]/2 - lane_mid) * (us_line_wide/img.shape[1])
        output = draw_text(output,"Radius of curvature: {0:.2f}m".format(curvature),1)
        if(veh_pos > 0):
            output = draw_text(output,"Vehicle is : {0:.2f}m to the left".format(abs(veh_pos)),2)
        elif(veh_pos < 0):
            output = draw_text(output,"Vehicle is : {0:.2f}m to the right".format(abs(veh_pos)),2)
        else:
            output = draw_text(output,"Vehicle is on center of the road".format(abs(veh_pos)),2)
    else:
        output = img
    return output

def get_lane_mid():
    lane_mid = ((right_line.line_base_pos - left_line.line_base_pos)/2) + left_line.line_base_pos
    return lane_mid

def update_curvature():
    if(left_line.radius_of_curvature_value == right_line.radius_of_curvature_value == 0):
        left_line.update_curvature()
        right_line.update_curvature()
    if(len(left_line.radius_of_curvature) == 50):
        left_line.update_curvature()
        right_line.update_curvature()
        left_line.radius_of_curvature.clear()
        right_line.radius_of_curvature.clear()
    return max(left_line.radius_of_curvature_value,right_line.radius_of_curvature_value)

# ultil to draw text on image
def draw_text(img,text,line):
    output                 = np.copy(img)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    coordinate             = (30,(70 * line))
    fontScale              = 2
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(output,text,
        coordinate,
        font,
        fontScale,
        fontColor,
        lineType)
    return output

#ultil to draw rectangle on image
def draw_rectangle(window_coordinate, overlay_image):
    for no in range (len(window_coordinate)):
        # Draw the windows on the visualization image
        coordinate = tuple(window_coordinate[no])
        cv2.rectangle(overlay_image, coordinate[0], coordinate[1], (0,255,0), 2)
    return overlay_image

# draw lane area
def update_lane_detection(left_line, right_line, Minv, binary_warped, undist):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.bestx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.bestx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result

def feed_video_through_pipline(video_path):
    outclip = 'output_images/' + video_path
    #clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,1)
    project_video = VideoFileClip(video_path)
    clip = project_video.fl_image(detect_lane)
    clip.write_videofile(outclip, audio=False)
    return

feed_video_through_pipline("project_video.mp4")
if (os.path.isfile("debug.txt")):
    os.remove("debug.txt")
#feed_video_through_pipline("challenge_video.mp4")
'''
series = glob.glob("test_images/straight_lines1.jpg")
for fname in series:
    image = cv2.imread(fname)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    sanity_check_save_image(image, detect_lane,"output_images/straight_lines1.jpg")
'''
