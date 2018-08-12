import numpy as np
import cv2
from matplotlib import pyplot as plt
class window():
    def __init__(self, nwindow, margin, mpixel, centroid):
        self.nwindow = nwindow
        self.margin = margin
        self.mpixel = mpixel
        self.centroid = centroid

    # Function to return the point within boundarys of widown-n
    # -- Input:
    # 1. window_no: number of widown - should be < than nwindow
    # 2. basex: base x to draw the rectangle (will be +- margin to get high & low)
    # 3. binary_wrapped: binary image to detect point within boundary
    # -- Output:
    # 1. point_inds: array of all detected points
    # 2. ncount: number of points
    # 3. window_coordinate: window coordinate to draw rectangle if requires
    def get_points_within_boundary(self, window_no, binary_wrapped):
        if (window_no < self.nwindow):
            # calculate window size
            window_height = np.int(binary_wrapped.shape[0]//self.nwindow)
            # calculate window coordinates
            win_y_low = int(binary_wrapped.shape[0] - (window_no+1) * window_height)
            win_y_high = int(binary_wrapped.shape[0] - window_no * window_height)
            win_x_low = int(self.centroid - self.margin)
            win_x_high = int(self.centroid + self.margin)

            window_coordinate = [(win_x_low, win_y_low), (win_x_high, win_y_high)]
            # Identify the x and y positions of all nonzero pixels in the image
            nonzero = binary_wrapped.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Get the indexes of point within boundary
            point_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
            # Get the count of point within boundary
            ncount = len(point_inds)
            # Update centroid of window
            if (ncount > self.mpixel):
                self.centroid = np.mean(nonzerox[point_inds])
        else:
            print ("Invalid window_no %d", (widown_no))

        return point_inds, window_coordinate, nonzerox, nonzeroy

    #Function to draw the rectagle and overlay that to image for sanity check
    def draw_rectangle(self, window_coordinate, overlay_image):
        # Draw the windows on the visualization image
        window_coordinate = tuple(window_coordinate)
        print (window_coordinate)
        output = np.copy(overlay_image)
        cv2.rectangle(output, window_coordinate[0], window_coordinate[1], (0,255,0), 2)
        return output

class Line():
    def __init__(self, nwindow, margin, mpixel, centroid, us_line_long, us_line_wide):
        self.us_line_wide = us_line_wide
        self.us_line_long = us_line_long

        self .has_data = False
        # was the line detected in the last iteration?
        self.detected = False
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = np.array([0,0,0], dtype='float')  
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        self.current_fit_cr = []
        #radius of curvature of the line in some units
        self.radius_of_curvature = [] 

        self.radius_of_curvature_value = 0
        #distance in meters of vehicle center from the line
        self.line_base_pos = 0
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # window used for running itegration
        self.window = window(nwindow, margin, mpixel, centroid)

        self.count = 0

        self.sliding_window =[]

        self.epsilon = float(4 * (10 ** (-3)))

        self.epsilon2 = float(100)

        self.epsilon3 = float(500)

    def update_curvature(self):
        self.radius_of_curvature_value = np.average(self.radius_of_curvature)

    def update_polycoeff(self, binary_warped):
        lane_point_inds = []
        ym_per_pix = self.us_line_long/720
        xm_per_pix = self.us_line_wide/640
        debug_image = np.copy(binary_warped)
        debug = []
        for no in range(self.window.nwindow):
            point_inds, window_coordinate, nonzerox, nonzeroy = self.window.get_points_within_boundary(no, binary_warped)
            self.sliding_window.append(window_coordinate)
            lane_point_inds.append(point_inds)
        # append all the x & y to Line
        ycal = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        if (len(lane_point_inds)):
            ## TODO: Need to add validation before updating lines
            try:
                lane_point_inds = np.concatenate(lane_point_inds)
            except ValueError:
                pass

            if(len(nonzerox[lane_point_inds]) & len(nonzeroy[lane_point_inds])):
                self.allx = nonzerox[lane_point_inds]
                self.ally = nonzeroy[lane_point_inds]
                var_current_fit = np.polyfit(self.ally, self.allx, 2)
                self.current_fit_cr = np.polyfit((np.array(self.ally) * ym_per_pix), (np.array(self.allx) * xm_per_pix), 2)
                self.detected = True
            else:
                self.detected = False

            if(self.detected == True):
                if(self.has_data == True):
                    self.diffs = np.absolute(var_current_fit - self.best_fit)
                    if ((self.diffs[0] < self.epsilon) & (self.diffs[1] < self.epsilon2)):
                        if (self.diffs[2] > self.epsilon3):
                            var_current_fit[2] = self.best_fit[2]
                        if(self.count == 10):
                            self.count = 0
                        if(len(self.current_fit) > self.count):
                            self.current_fit.pop(self.count)
                        self.current_fit.insert(self.count, var_current_fit)
                        self.best_fit = np.average(self.current_fit, axis=0)
                        self.radius_of_curvature.append(((1 + (2*self.current_fit_cr[0]*max(self.ally)*ym_per_pix + self.current_fit_cr[1])**2)**1.5) / np.absolute(2*self.current_fit_cr[0]))
                    else:
                        self.detected = False
                else:
                    self.current_fit.insert(self.count, np.polyfit(self.ally, self.allx, 2))
                    self.best_fit = self.current_fit[-1]
                    self.radius_of_curvature.append(((1 + (2*self.current_fit_cr[0]*max(self.ally)*ym_per_pix + self.current_fit_cr[1])**2)**1.5) / np.absolute(2*self.current_fit_cr[0]))
                self.count = self.count + 1
                self.has_data = True
                try:
                    self.bestx = self.best_fit[0]*ycal**2 + self.best_fit[1]*ycal + self.best_fit[2]
                    self.line_base_pos = self.bestx[-1]
                except TypeError:
                    # Avoids an error if `left` and `right_fit` are still none or incorrect
                    print('The function failed to fit a line!')
                    self.bestx  = 1*ycal**2 + 1*ycal
                print("+++++++++++", file=open("debug.txt", "a"))
                print (self.diffs,file=open("debug.txt", "a"))
                print("----", file=open("debug.txt", "a"))
                print (self.best_fit,file=open("debug.txt", "a"))

            




