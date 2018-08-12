
# Advanced Lane Finding Project

The steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image01]: ./diagram/flow_diagram.png
[image02]: ./output_images/chessboard.png
[image03]: ./output_images/color_grd_thresh.png "Color & gradient thresholds"
[image04]: ./output_images/warped.png "Waped Images"
[image05]: ./output_images/sliding_window.png "Sliding Window"
[image06]: ./output_images/straight_lines1.jpg "Straight Line"
[image07]: ./output_images/undistort.png "Undistort image"
[video1]: ./project_video.mp4 "Video"

---
## How to start
The project is done within 2 main file which are:
- main.py: main class & entry point of the project
- line class: helper class to store and process data for left & right Lane

This project does not use IPython notebook since I am not really get into it, and I think it does not cause any issue, in fact, it could be easier for you to run this project

## Flow diagram of project

In order to make life easy, I have create a simple flow diagram for this project to help people better understanding my program :simple_smile:

![Flow Diagram][image01]

## Camera Calibration

#### 1. Camera calibration

The code for this step is contained in the second code cell of the `main.py` located in root folder with function name `camera_calibration()`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

![chessboard][image02]

I then save the output `objpoints` and `imgpoints` to pickle file as dictionary so that later in this project, it can be loaded for undistorting the processing image with function call `cal_undistort()`, which is also defined in `main.py`

#### 2. Undistort image

The code for this step is contained in the third code cell of the `main.py` located in root folder with function name `cal_undistort()`

I start by loading the `calibrate.pickle` to `objpoints` and `imgpoints` and use those data to calibrate the camera and calling `undistort` to get the output image as following

![Undistort example][image07]

### Pipeline (single images)

#### 1. Create binary image by applying color & gradient thresholds & extract ROI

Firstly, I apply the sobelx to compute the gradient with thresholds `sx_thresh=(20, 100)`. In order to increase the confidence of detection left lane & right lane, I have also combine the color thresholds for green channel of RGB because it give better information of detecting white lane & s channel of HLS since it give best information of detecting yellow lane `g_thresh=(170,255)`, `s_thresh=(150, 190)`, `h_thresh = (15,20)`. The combined binary image is calculated as following: `combined_binary[((s_binary == 1) & (h_binary == 1) & (sxbinary == 1)) | ((g_binary == 1) & (sxbinary == 1))] = 1`

![Combined binary][image03]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 196 through 213 in the file `main.py`.  The `perspective_transform()` function takes as inputs an image (`img`), and output as `warped_img`, `original_img` (for debugging purpose) & `Minv`  I chose the hardcode the source and destination points in the following manner:

| Source        | Destination   |
|:-------------:|:-------------:|
| 250, 678      | 320, 720      |
| 585, 456      | 320, 0        |
| 700, 456      | 960, 0        |
| 1045, 678     | 960, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warpPerspective image][image04]

#### 4.Identifying lane-line pixels and fit their positions with a polynomial?

After getting the warped image, I apply use sliding window to detect the lane line base. The left line and the right line is differentiated base on the centroid, left liane is initiate as following:
```python
left_centroid = np.argmax(histogram[:midpoint])
right_centroid = np.argmax(histogram[midpoint:]) + midpoint
  ```
The code for those task includes 2 function call `update_polycoeff` and `get_points_within_boundary` in the ultility file `line_class.py`. In order not to update the polyfit for some scene which are not detected any line, also eliminate the wrong detection. I have also include simple sanity check as following:
```python
if ((np.absolute(self.diffs[0]) < self.epsilon)):
  if(self.count == 6):
      self.count = 0
  if(len(self.current_fit) > self.count):
      self.current_fit.pop(self.count)
  self.current_fit.insert(self.count, np.polyfit(self.ally, self.allx, 2))
  self.best_fit = np.average(self.current_fit, axis=0)
  self.radius_of_curvature.append(((1 + (2*self.current_fit_cr[0]*max(self.ally)*ym_per_pix + self.current_fit_cr[1])**2)**1.5) / np.absolute(2*self.current_fit_cr[0]))
```

The `best_fit` polynomial is calculated as average of last 6 detected frame in the video so that the output could be a little bit smoother :simple_smile:

![alt text][image05]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to center.

This is quite simple to calculate the curvature base on the polyfit, but the only things that need to pay attention is that we need to consider about the scale of image to real world which is done by:
```python
ym_per_pix = self.us_line_long/720
xm_per_pix = self.us_line_wide/640
```

The number `720` and `640` is the part of image we transfomr to bird-eyes view. As shown in image above, we could see that the area of transforming is from `(320,960)` for width and `(0,720)` for height.

Later, the curvature is calculate be first getting the polyfit which scaling to real world dimension and then calculate the radius of curvature by applying the formula descriped [here](https://www.intmath.com/applications-differentiation/8-radius-curvature.php):
```python
self.current_fit_cr = np.polyfit((np.array(self.ally) * ym_per_pix), (np.array(self.allx) * xm_per_pix), 2)
...
self.radius_of_curvature.append(((1 + (2*self.current_fit_cr[0]*max(self.ally)*ym_per_pix + self.current_fit_cr[1])**2)**1.5) / np.absolute(2*self.current_fit_cr[0]))
...
self.radius_of_curvature_value = np.average(self.radius_of_curvature)
```

#### 6. Final Result :simple_smile:

As shown in the flow diagram above, all of the step is done by simple function `detect_lane` which is defined in `main.py` which will be use to feed video as well. Kindly refer the the result image as following:

![alt text][image06]

---

### Pipeline (video)

I have apply this pipeline for video `project_video.mp4` in root folder and the result is some how looks nice to me, although there is a very very very long way to make this algorithm works under real world!!

Here's a [link to my video result](./output_images/project_video.mp4)

---

### Discussion

#### Known issue & problem

- This algorithm use too much fixed setting for `perspective_transform` & `color_gradient_filter` which resulting in lacking of flexibility hence it could not be applied to real world, but just for project video, it seems to be working very well at least :) (of course)
- ROI is also fixed area, I could not success to make it dynamic base on vanishing point.
- This algorithm is working very slow, hence it could not be applied to real time video. I think converted to C++ program could improve processing time, but to make it ready for real time video, I strongly believe that we have to swtich to use parallel computing or GPU computing.
- The constrast of video is very much affected to the quality of the detection, but in real world, the line could be no so clear like in project video, we need to have better approach to overcome this issue.
