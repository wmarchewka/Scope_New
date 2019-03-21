import cv2
import time
import sys
import numpy as np
import cvui
import math

from simple_pid import PID
from multiprocessing import Process

class Scope(object):

    calibration_ok: bool

    def __init__(self):

        #
        self.data = {'points': [], 'linelengths': []}  # dictionary to store all data
        self.lines_found = []

        #camera
        self.mainframe = None
        self.camera_capture_width = 800
        self.camera_capture_height = 600
        self.color_black = (0, 0, 0)
        self.cam_frame = self.frame_create(self.camera_capture_height, self.camera_capture_width, self.color_black)
        self.camera_create()
        self.grabbed = None

        #contours
        self.contour_frame = None
        self.contours_found_hierarchy = None
        self.focus_value = 0
        self.contours_found = None
        self.cal_line_distance_avg = 0
        self.calibration_ok = False
        self.contours_found_quan = None

        # main window
        self.mainwindow_name = "main"
        self.mainwindow_x = 2100
        self.mainwindow_y = 0
        self.mainwindow_width = 800
        self.mainwindow_height = 900
        self.color_black = (0, 0, 0)

        # aux window
        self.auxwindow_name = 'aux'
        self.auxwindow_width = 300
        self.auxwindow_height = 300
        self.auxwindow_x = 2950
        self.auxwindow_y = 0

        # create main frame and window
        self.mainframe = self.frame_create(self.mainwindow_width, self.mainwindow_height, self.color_black)
        self.window_create(self.mainwindow_name, self.mainwindow_x, self.mainwindow_y)

        # create aux frame and window
        auxframe = self.frame_create(self.auxwindow_width, self.auxwindow_height, self.color_black)
        self.window_create(self.auxwindow_name, self.auxwindow_x, self.mainwindow_y)

        # button values
        self.ortho_button_text = "ORTHO OFF"
        self.autotune_button_text = "AUTO TUNE OFF"
        self.focus_button_text = "FOCUS START"
        self.cal_button_text = "CAL HOLD"

        self.ortho_mode = False
        self.start_focus_mode = False
        self.autotune_mode = False
        self.cal_mode = False

        # trackbar
        self.canny_trackbar_max_value = 100
        self.blur_trackbar_max_value = 50
        self.blur_value = [7]
        self.canny_value = [20]
        self.calibration_ok = False
        self.cal_hold = False

        self.contour_frame_start_x = 0
        self.contour_frame_end_x = 450
        self.contour_frame_start_y = 500
        self.contour_frame_end_y = 600
        self.contour_area = [self.contour_frame_start_x, self.contour_frame_end_x,
                             self.contour_frame_start_y, self.contour_frame_end_y]

        # mouse
        self.camcapture_height = 0
        self.camcapture_width = 0
        self.actual_mm_x = 0
        self.actual_mm_y = 0
        self.px_mm_conversion = 0
        self.mouse_found_line = None
        self.line_draw_active = False
        self.mouse_color = (0, 0, 0)

        self.grabbed_cal_value = False

        # self.p1 = Process(target=self.camera_read, args=())
        # self.p1.start()
        # self.p1.join()

    ####################################################################################
    def shutdown(self):
        self.p1.kill()
        cv2.destroyAllWindows()
        sys.exit()

    ####################################################################################
    def update(self):

        self.mainframe = self.frame_create(self.mainwindow_width, self.mainwindow_height, self.color_black)
        self.camera_read()
        self.buttons_create_monitor()
        self.trackbars_create_monitor()
        self.mouse_check()
        self.contours_frame_prepare()
        self.contours_search()
        self.contours_draw()
        self.contours_calibration_check()
        self.lines_draw()
        self.roi_draw()
        self.camera_frame_attach()
        self.windows_show()
        self.keypress_check()

        ####################################################################################

    def camera_create(self):
        # create camera instance
        self.camcapture_instance = 1
        self.camcapture = cv2.VideoCapture(self.camcapture_instance)
        self.camcapture.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_capture_width)
        self.camcapture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_capture_height)
        self.camcapture.set(cv2.CAP_PROP_FPS, 100)

        ####################################################################################

    def camera_read(self):
        (self.grabbed, temp_cam_frame) = self.camcapture.read()
        # flip frame to the correct orientation
        self.cam_frame = cv2.flip(temp_cam_frame, -1)

        ####################################################################################

    def frame_create(self, camcapture_height, camcapture_width, color):
        frame = np.zeros((camcapture_height, camcapture_width, 3), np.uint8)
        frame[:] = color
        return frame

    ####################################################################################
    def contours_frame_prepare(self):

        gray_frame = cv2.cvtColor(self.cam_frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray_frame, (int(self.blur_value[0]), int(self.blur_value[0])), 0)
        blurred_frame_roi = blurred_frame[self.contour_area[2]:self.contour_area[3],
                            self.contour_area[0]:self.contour_area[1]]
        self.focus_value = cv2.Laplacian(blurred_frame_roi, cv2.CV_64F).var()
        self.focus_value = self.focus_value - 2
        self.focus_value = (self.focus_value / 15) * 100
        canny_frame = cv2.Canny(blurred_frame_roi, self.canny_value[0], self.canny_value[0] * 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated_frame = cv2.dilate(canny_frame, kernel)
        self.contour_frame = dilated_frame
        self.contours_search()

    ####################################################################################
    def contours_search(self):
        self.contours_found, self.contours_found_hierarchy = cv2.findContours(self.contour_frame,
                                                                              cv2.RETR_EXTERNAL,
                                                                              cv2.CHAIN_APPROX_SIMPLE)
        self.contours_found.sort(key=lambda c: np.min(c[:, :, 0]))
        self.contours_found_quan = len(self.contours_found)
        self.contours_draw()

    ###############################################################################
    def contours_draw(self):

        frame_start_y = self.contour_area[2]
        frame_end_y = self.contour_area[3]

        offset = (0, 500)
        cnt = 0
        for c in self.contours_found:
            cv2.drawContours(self.cam_frame, self.contours_found, cnt, (255, 0, 0), 1, cv2.LINE_8,
                             self.contours_found_hierarchy, 0, offset)
            cnt += 1
            ext_left = tuple(c[c[:, :, 0].argmin()][0])
            ext_right = tuple(c[c[:, :, 0].argmax()][0])
            contour_width = tuple(np.subtract(ext_right, ext_left))
            center_horz = int((ext_left[0] + (contour_width[0] / 2)))
            cv2.line(self.cam_frame, (center_horz, frame_end_y), (center_horz, frame_start_y), (0, 0, 255), 1)
            cv2.putText(self.cam_frame, str(center_horz), (center_horz, frame_start_y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, 1)
        self.contours_calibration_check()

    ###############################################################################
    def contours_calibration_check(self):
        # we now want to find the distance between the center of the first and next contour
        # make sure the contours are sorted from least to most across the X axis
        # keep a running total of found spaces and a running average
        cal_lines = 0
        self.cal_line_distance_total = 0
        self.cal_line_distance_avg = 0
        # TODO: make this without loop
        last_center_horz = 0
        if len(self.contours_found) == 11:
            for c in self.contours_found:
                ext_left = tuple(c[c[:, :, 0].argmin()][0])
                ext_right = tuple(c[c[:, :, 0].argmax()][0])
                contour_width = tuple(np.subtract(ext_right, ext_left))
                center_horz = int((ext_left[0] + (contour_width[0] / 2)))
                diff = abs(last_center_horz - center_horz)
                cal_lines += 1
                if 2 < diff < 200:
                    if cal_lines > 1:
                        self.cal_line_distance_total = self.cal_line_distance_total + diff
                last_center_horz = center_horz
            if self.cal_line_distance_total is not 0:
                self.cal_line_distance_total = self.cal_line_distance_total / (cal_lines - 1)
                self.calibration_ok = True
                self.px_mm_conversion = 1 / self.cal_line_distance_total
        else:
            self.calibration_ok = False
            cv2.displayOverlay(self.mainwindow_name, "Please complete calibration", 1000)
        if self.cal_mode:
            if self.grabbed_cal_value is False:
                self.hold_px_mm_conversion = self.px_mm_conversion
                self.hold_cal_line_distance_total = self.cal_line_distance_total
                self.grabbed_cal_value = True
            else:
                self.px_mm_conversion = self.hold_px_mm_conversion
                self.cal_line_distance_total = self.hold_cal_line_distance_total
        else:
            pass

        self.lines_draw()

    ###############################################################################
    def lines_draw(self):
        font_scale = 0.3
        font_thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        line = 0
        num = 0
        number_of_points = len(self.data['points'])

        while num < (number_of_points - 1):
            if number_of_points > 1:
                line = line + 1
                start_data = self.data['points'][num]
                end_data = self.data['points'][num + 1]
                start_data_x = self.data['points'][num][0]
                end_data_x = self.data['points'][num + 1][0]
                start_data_y = self.data['points'][num][1]
                line_length_data = abs((np.subtract(self.data['points'][num + 1], self.data['points'][num])))
                if self.mouse_found_line == line:
                    line_color = (0, 255, 0)
                else:
                    line_color = (0, 0, 255)
                cv2.line(self.cam_frame, start_data, end_data, line_color, 2)
                # TODO: MAKE SMALL LINES IF LINE IS DRAWN IN Y DIMENSION
                # we have to determine whether this line was draw in the X or Y
                # TODO: combine functions for this code
                if line_length_data[0] > line_length_data[1]:
                    x_length_pixels = line_length_data[0]
                    # y_length_pixels = line_length_data[1]
                    # determine length of line in mm both X and Y
                    x_length_calibrated = x_length_pixels * self.px_mm_conversion
                    if start_data_x < end_data_x:
                        start_x = end_data_x - (x_length_pixels / 2)
                    else:
                        start_x = start_data_x - (x_length_pixels / 2)
                    cv2.line(self.cam_frame, (start_data[0], start_data[1] - 5), (start_data[0], start_data[1] + 5),
                             line_color, 2)
                    cv2.line(self.cam_frame, (end_data[0], end_data[1] - 5), (end_data[0], end_data[1] + 5),
                             line_color, 2)
                    string_data = "(Line:{:2d}) ({:4.3f}mm)".format(line, x_length_calibrated)
                    label_width, label_height = cv2.getTextSize(string_data, font, font_scale, font_thickness)[0]
                    start_x = int(start_x - (label_width / 2))
                    # figure out the start Y
                    start_y = int(start_data_y) - 7
                    # place text on line
                    cv2.putText(self.cam_frame, string_data, (start_x, start_y), font, font_scale, line_color,
                                font_thickness,
                                cv2.LINE_AA)
                    line_length_final = x_length_calibrated
                    # store data in array
                    self.data['linelengths'].insert(line, [line_length_final])
                else:
                    # print("line drawn in y direction")
                    # same as above
                    x_length_pixels = line_length_data[0]
                    y_length_pixels = line_length_data[1]
                    # x_length_calibrated = x_length_pixels * self.px_mm_conversion
                    y_length_calibrated = y_length_pixels * self.px_mm_conversion
                    start_y = start_data_y + (y_length_pixels / 2)
                    string_data = "(Line:{:2d})({:4.3f}mm)".format(line, y_length_calibrated)
                    label_width, label_height = cv2.getTextSize(string_data, font, font_scale, font_thickness)[0]
                    start_y = int(start_y - (label_width / 2))
                    start_x = int(start_data_x)
                    cv2.line(self.cam_frame, (start_data[0] - 5, start_data[1]), (start_data[0] + 5, start_data[1]),
                             line_color, 2)
                    cv2.line(self.cam_frame, (end_data[0] - 5, end_data[1]), (end_data[0] + 5, end_data[1]),
                             line_color, 2)
                    cv2.putText(self.cam_frame, string_data, (start_x, start_y), font, font_scale, line_color,
                                font_thickness,
                                cv2.LINE_AA)
                    line_length_final = y_length_calibrated

                    self.data['linelengths'].insert(line, [line_length_final])
            # increase line count
            num = num + 2

    ###############################################################################

    def line_iterator(self, p1, p2, x, y):
        # this routine takes two points and a x and y value from mouse and creates all points
        # from those two points and figure out if mouse is on one of those points and returns the line
        # TODO try redesign so that all the lines can be checked without looping
        p1_x = p1[0]
        p1_y = p1[1]
        p2_x = p2[0]
        p2_y = p2[1]

        # difference and absolute difference between points
        # used to calculate slope and relative location between points
        d_x = p2_x - p1_x
        d_y = p2_y - p1_y
        d_xa = np.abs(d_x)
        d_ya = np.abs(d_y)

        # predefine numpy array for output based on distance between points
        itbuffer = np.empty(shape=(np.maximum(d_ya, d_xa), 2))
        itbuffer.fill(np.nan)

        # Obtain coordinates along the line using a form of Bresenham's algorithm
        neg_y = p1_y > p2_y
        neg_x = p1_x > p2_x
        if p1_x == p2_x:  # vertical line segment
            itbuffer[:, 0] = p1_x
            if neg_y:
                itbuffer[:, 1] = np.arange(p1_y - 1, p1_y - d_ya - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(p1_y + 1, p1_y + d_ya + 1)
        elif p1_y == p2_y:  # horizontal line segment
            itbuffer[:, 1] = p1_y
            if neg_x:
                itbuffer[:, 0] = np.arange(p1_x - 1, p1_x - d_xa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(p1_x + 1, p1_x + d_xa + 1)
        else:  # diagonal line segment
            steep_slope = d_ya > d_xa
            if steep_slope:
                # had to replace the as written slope line with my own. not sure why they were
                # doing the way they did
                # slope = dX.astype(np.float32) / dY.astype(np.float32)
                slope = d_x / d_y
                if neg_y:
                    itbuffer[:, 1] = np.arange(p1_y - 1, p1_y - d_ya - 1, -1)
                else:
                    itbuffer[:, 1] = np.arange(p1_y + 1, p1_y + d_ya + 1)
                itbuffer[:, 0] = (slope * (itbuffer[:, 1] - p1_y)).astype(np.int) + p1_x
            else:
                # slope = dY.astype(np.float32) / dX.astype(np.float32)
                slope = d_y / d_x
                if neg_x:
                    itbuffer[:, 0] = np.arange(p1_x - 1, p1_x - d_xa - 1, -1)
                else:
                    itbuffer[:, 0] = np.arange(p1_x + 1, p1_x + d_xa + 1)
                itbuffer[:, 1] = (slope * (itbuffer[:, 0] - p1_x)).astype(np.int) + p1_y

        # Remove points outside of image
        col_x = itbuffer[:, 0]
        col_y = itbuffer[:, 1]

        itbuffer = itbuffer[
            (col_x >= 0) & (col_y >= 0) & (col_x < self.camera_capture_width) & (
                        col_y < self.camera_capture_height)]
        # foundItem = np.isin(itbuffer, [x, y])
        if x in itbuffer[:, [0]] and y in itbuffer[:, [1]]:
            return True
        else:
            return False

    ####################################################################################
    def roi_draw(self):
        cv2.rectangle(self.cam_frame, (self.contour_frame_start_x, self.contour_frame_start_y),
                      (self.contour_frame_end_x, self.contour_frame_end_y), (255, 0, 0), 3)

    ####################################################################################
    def camera_frame_attach(self):
        cvui.image(self.mainframe, 0, 0, self.cam_frame)

    ####################################################################################

    def window_create(self, name, x, y):
        cvui.init(name)
        # cvui.window(image, x, y, w, h, title)
        cv2.moveWindow(name, x, y)

    ####################################################################################
    def frame_create(self, w=0, h=0, color=(0, 0, 0)):
        frame = np.zeros((h, w, 3), np.uint8)
        frame[:] = color
        return frame

    ####################################################################################

    def trackbars_create_monitor(self):
        # canny trackbar
        cvui.trackbar(self.mainframe, 0, 650, 400, self.canny_value, 0, self.canny_trackbar_max_value, 5,
                      '%.0Lf', cvui.TRACKBAR_HIDE_STEP_SCALE)
        # blur trackbar
        cvui.trackbar(self.mainframe, 0, 700, 400, self.blur_value, 1, self.blur_trackbar_max_value, 5,
                      '%.0Lf', cvui.TRACKBAR_DISCRETE, 2)

        ####################################################################################

    def buttons_create_monitor(self):

        ortho_button_press = cvui.button(self.mainframe, 450, 650, 100, 30, self.ortho_button_text)
        if ortho_button_press and self.ortho_mode == False:
            self.ortho_mode = True
            self.ortho_button_text = 'ORTHO ON'
        elif ortho_button_press and self.ortho_mode == True:
            self.ortho_mode = False
            self.ortho_button_text = 'ORTHO OFF'

        autotune_button_press = cvui.button(self.mainframe, 450, 690, 100, 30, self.autotune_button_text)
        if autotune_button_press and self.autotune_mode == False:
            self.autotune_mode = True
            self.autotune_button_text = 'AUTO TUNE OFF'
        elif autotune_button_press and self.autotune_mode == True:
            self.autotune_mode = False
            self.autotune_button_text = 'AUTO TUNE ON'

        focus_button_press = cvui.button(self.mainframe, 570, 650, 100, 30, self.focus_button_text)
        if focus_button_press and self.start_focus_mode == True:
            self.start_focus_mode = False
            self.focus_button_text = 'FOCUS START'
        elif focus_button_press and self.start_focus_mode == False:
            self.start_focus_mode = True
            self.focus_button_text = 'FOCUS STOP'

        cal_button_press = cvui.button(self.mainframe, 570, 690, 100, 30, self.cal_button_text)
        if cal_button_press and self.cal_mode == True:
            self.cal_mode = False
            self.cal_button_text = 'CAL HOLD'
        elif cal_button_press and self.cal_mode == False:
            self.cal_mode = True
            self.cal_button_text = 'CAL HOLDING'

    ####################################################################################
    def windows_show(self):
        cvui.update()
        cvui.context(self.mainwindow_name)
        cvui.imshow(self.mainwindow_name, self.mainframe)
        cv2.displayStatusBar(self.mainwindow_name,
                             "{:3.2f}pmm X:{:03d} Y:{:03d} Focus:{:05d} Contours:{:02d} Xmm:{:7.2f}  Ymm{:7.2f} Color:{}"
                             .format(self.cal_line_distance_total, self.mouse_x, self.mouse_y,
                                     int(self.focus_value),
                                     self.contours_found_quan,
                                     self.actual_mm_x,
                                     self.actual_mm_y,
                                     str(self.mouse_color)), 0)

    ####################################################################################
    def keypress_check(self):

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            self.shutdown()
        elif key & 0xFF == 27:
            print("Escape key")
            if self.line_draw_active:
                current_point_number = len(self.data['points']) - 1
                print("Deleting current point " + str(current_point_number))
                del self.data['points'][current_point_number]
                self.line_draw_active = False



 ####################################################################################
    def mouse_check(self):
        self.mouse_y = cvui.mouse().y
        self.mouse_x = cvui.mouse().x
        if self.mouse_in_range():
            if cvui.mouse(cvui.DOWN):
                self.mouse_down()
            else:
                self.mouse_move()

    ####################################################################################
    def mouse_in_range(self):
        # ensure that the self.mouse_x and self.mouse_y of the mouse do not cross the video boundaries
        if (self.mouse_x > 0) and (self.mouse_x < self.camera_capture_width - 1) and (self.mouse_y > 0) and \
                (self.mouse_y < self.camera_capture_height - 1):
            return True
        else:
            return False

    ####################################################################################
    def mouse_line_length(self):
        self.mouse_line_length_x = abs(
            (self.mouse_first_point_x - self.mouse_x) * self.px_mm_conversion)
        self.mouse_line_length_y = abs(
            (self.mouse_first_point_y - self.mouse_y) * self.px_mm_conversion)

    ####################################################################################
    def mouse_ortho_mode(self):
        if self.ortho_mode:
            self.mouse_line_length()
            if self.mouse_line_length_x > self.mouse_line_length_y:
                self.mouse_y = self.mouse_first_point_y
            if self.mouse_line_length_y > self.mouse_line_length_x:
                self.mouse_x = self.mouse_first_point_x

    ####################################################################################
    def mouse_delete_point(self):
        # print("Clk on line " + str(glbFoundLine))
        del self.data['points'][(self.mouse_found_line * 2) - 1]
        del self.data['points'][(self.mouse_found_line * 2) - 2]
        self.mouse_found_line = None

    ####################################################################################
    def mouse_move(self):
        self.mouse_get_color()
        self.mouse_draw_crosshair()
        if self.calibration_ok:
            self.actual_mm_x = self.px_mm_conversion * self.mouse_x
            self.actual_mm_y = self.px_mm_conversion * self.mouse_y
            if self.line_draw_active:
                if self.ortho_mode: self.mouse_ortho_mode()
                cv2.line(self.cam_frame, (self.mouse_first_point_x, self.mouse_first_point_y),
                         (self.mouse_x, self.mouse_y), (0, 0, 255), 1)
                self.mouse_line_length()
                theta_radians = math.atan2(self.mouse_x - self.mouse_first_point_x,
                                           self.mouse_y - self.mouse_first_point_y)
                line_angle_deg = math.degrees(theta_radians)
                cv2.displayOverlay(self.mainwindow_name,
                                   "self.mouse_x length={:4.3f}mm   self.mouse_y length={:4.3f}mm   Angle={:3.1f}deg".format(
                                       self.mouse_line_length_x,
                                       self.mouse_line_length_y, line_angle_deg), 1000)
            else:
                self.mouse_on_line()

    ####################################################################################
    def mouse_draw_crosshair(self):
        cv2.line(self.cam_frame, (self.mouse_x + 15, self.mouse_y), (self.mouse_x - 15, self.mouse_y), (0, 0, 255), 1)
        cv2.line(self.cam_frame, (self.mouse_x, self.mouse_y - 15), (self.mouse_x, self.mouse_y + 15), (0, 0, 255), 1)

    ####################################################################################
    def mouse_down(self):
        if self.mouse_found_line is not None:
            self.mouse_delete_point()
        elif self.line_draw_active:
            self.line_draw_active = False
            local_point_number = len(self.data['points'])
            if self.ortho_mode: self.mouse_ortho_mode()
            self.data['points'].insert(local_point_number, (self.mouse_x, self.mouse_y))
        else:
            self.line_draw_active = True
            local_point_number = len(self.data['points'])
            self.data['points'].insert(local_point_number, (self.mouse_x, self.mouse_y))
            self.mouse_first_point_x = self.mouse_x
            self.mouse_first_point_y = self.mouse_y
            print("First point")
    ####################################################################################
    def mouse_on_line(self):
        line = 0
        num = 0
        number_of_points = len(self.data['points'])

        # TODO: come up with better solution
        while True:
            if number_of_points > 1:
                line = line + 1
                start_data = self.data['points'][num]
                end_data = self.data['points'][num + 1]
                return_val = self.line_iterator(start_data, end_data, self.mouse_x, self.mouse_y)
                if return_val:
                    #scope.lines_found[line] = True
                    self.mouse_found_line = line
                    print("found line" + str(line))
                    break
                else:
                    self.mouse_found_line = None
            num = num + 2
            if num >= number_of_points - 1:
                break

    ####################################################################################
    def mouse_get_color(self):
        if self.mouse_in_range():
            mouse_color_frame = self.cam_frame.copy()
            self.mouse_color = mouse_color_frame[self.mouse_y, self.mouse_x]
        else:
            self.mouse_color = (0, 0, 0)

def main():
    sc = Scope()
    last_time = 0
    print("Create class instance")
    while True:
        sc.update()
        elapsed_time = (time.time() - last_time) * 1000
        #print("Elapsed time ms = {:4.1f}".format(elapsed_time))
        last_time = time.time()

if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()