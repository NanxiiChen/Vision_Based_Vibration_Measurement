import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib.gridspec import GridSpec

class VisionVibrationMeasurement:
    """
    A class to perform vision-based vibration measurement by detecting markers in image sequences.
    It replicates the logic of the provided MATLAB script.
    """
    def __init__(self, img_dir, output_folder='result_py'):
        """
        Initializes the VisionVibrationMeasurement class.

        Args:
            img_dir (str): Path to the directory containing input JPG images.
            output_folder (str): Path to the directory where results (e.g., output video) will be saved.
        """
        # Configuration
        self.folderOut = output_folder
        if not os.path.exists(self.folderOut):
            os.makedirs(self.folderOut)

        self.DEBUG = False # Set to True for intermediate plots/debug info

        # Get a list of all jpg files in an image directory
        self.imgPathList = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
        if not self.imgPathList:
            # Try common image extensions if JPG not found
            for ext in ['*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']:
                self.imgPathList.extend(glob.glob(os.path.join(img_dir, ext)))
            self.imgPathList = sorted(list(set(self.imgPathList))) # Remove duplicates and sort

        if not self.imgPathList:
            raise FileNotFoundError(f"No suitable images (jpg, jpeg, png, bmp, tif) found in {img_dir}")

        self.nImg = len(self.imgPathList)

        # Image sizes (determine from first image)
        first_img_bgr = cv2.imread(self.imgPathList[0])
        if first_img_bgr is None:
            raise ValueError(f"Could not read the first image: {self.imgPathList[0]}")
        # OpenCV shape is (height, width, channels)
        self.imgSize = (first_img_bgr.shape[0], first_img_bgr.shape[1]) # (height, width)

        # Initial locations of target points (from MATLAB script, 1-based)
        # These are y-coordinates (height) and x-coordinates (width)
        # IMPORTANT: User must ensure these values are appropriate for their images.
        # The example values from MATLAB might be out of bounds for a 1300px high image.
        # The script handles out-of-bounds by clamping, which might lead to empty search strips.
        iniLocH_matlab = np.array([1904, 1420, 931, 471])  # y-coordinates (rows)
        # iniLocW_matlab = np.array([148, 310, 466, 625, 778, 940, 1111]) # x-coordinates (cols)
        iniLocW_matlab = np.array([148, 313, 472, 634, 790, 955, 1129]) # x-coordinates (cols)

        # Convert to 0-based indexing for Python
        self.iniLocH = iniLocH_matlab - 1
        self.iniLocW = iniLocW_matlab - 1
        
        self.targetStripH = 100 # Height of the horizontal strip for searching targets
        self.targetSize = 20    # Diameter for plotting initial circles (visual aid)

        if self.DEBUG:
            self._plot_initial_targets()

        # Point connection map for skeleton (0-based)
        # MATLAB ptH (horizontal connections within rows)
        ptH_matlab = []
        for i in range(4): # 4 rows of targets
            for j in range(6): # 6 connections per row (7 points)
                base = i * 7 + 1 # 1-based start index of current row
                ptH_matlab.append([base + j, base + j + 1])
        ptH_matlab = np.array(ptH_matlab)

        # MATLAB ptV (vertical connections between specific columns)
        # ptVtmp = [1 8; 8 15; 15 22]; (1st column of points)
        # ptV = [ptVtmp; ptVtmp+3; ptVtmp+6]; (1st, 4th, 7th columns of points)
        ptV_matlab_col1 = np.array([[1,8],[8,15],[15,22]])
        ptV_matlab_col4 = ptV_matlab_col1 + 3 # Points (4,11), (11,18), (18,25)
        ptV_matlab_col7 = ptV_matlab_col1 + 6 # Points (7,14), (14,21), (21,28)
        ptV_matlab = np.vstack((ptV_matlab_col1, ptV_matlab_col4, ptV_matlab_col7))
        
        self.ptMap = np.vstack((ptH_matlab, ptV_matlab)) - 1 # Convert to 0-based

        if self.DEBUG:
            self._plot_truss_skeleton()

        # Threshold values for target detection
        self.threshTarget = 0.1       # For image binarization (scaled to 0-255)
        self.targetAreaLB = 100       # Lower bound for target area
        self.targetAreaUB = 500       # Upper bound for target area
        self.targetLargeMovement = 100 # Max allowed pixel movement between frames
        # Circularity threshold: MATLAB's `abs(circularity-1)<1` means 0 < circularity < 2.
        # This is a very lenient check. `circThrsh = 0.8` from MATLAB script was unused in this condition.
        self.circularity_deviation_threshold = 1.0 

        # Morphological structuring element (disk of radius 3 -> diameter 7)
        self.se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*3+1, 2*3+1))

        # Video parameters
        self.camerafs = 20  # Frame rate
        self.filename_video = os.path.join(self.folderOut, 'vibration_py.mp4')

        # Storage for locations of targets: (num_targets, 2_coords (x,y), num_images)
        self.num_targets = len(self.iniLocH) * len(self.iniLocW) # Should be 4*7=28
        self.LocPt = np.zeros((self.num_targets, 2, self.nImg))

        # For combined plot figure used in video frames
        self.fig_plot_combined = None
        
        # Precompute initial XY coordinates for all 28 points (0-based)
        # Order: (W0,H0), (W1,H0)...(W6,H0), (W0,H1), (W1,H1)...
        self.initial_points_xy = np.zeros((self.num_targets, 2))
        pts_x_init = np.tile(self.iniLocW, len(self.iniLocH))
        pts_y_init = np.repeat(self.iniLocH, len(self.iniLocW))
        for i in range(self.num_targets):
            self.initial_points_xy[i, 0] = pts_x_init[i]
            self.initial_points_xy[i, 1] = pts_y_init[i]


    def _plot_initial_targets(self):
        """Helper to plot initial target locations if DEBUG is True."""
        img_bgr = cv2.imread(self.imgPathList[0])
        if img_bgr is None: return

        img_display = img_bgr.copy()
        for i in range(self.num_targets):
            center_x = int(self.initial_points_xy[i, 0])
            center_y = int(self.initial_points_xy[i, 1])
            # Ensure points are within image bounds for drawing
            if 0 <= center_x < self.imgSize[1] and 0 <= center_y < self.imgSize[0]:
                cv2.circle(img_display, (center_x, center_y), int(self.targetSize / 2), (0, 0, 255), 2)

        plt.figure()
        plt.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB))
        plt.title("Initial Target Locations (0-based)")
        plt.show()

    def _plot_truss_skeleton(self):
        """Helper to plot initial truss skeleton if DEBUG is True."""
        plt.figure()
        for i in range(self.ptMap.shape[0]):
            p1_idx, p2_idx = self.ptMap[i, 0], self.ptMap[i, 1]
            # Ensure indices are valid
            if 0 <= p1_idx < self.num_targets and 0 <= p2_idx < self.num_targets:
                pt1_x, pt1_y = self.initial_points_xy[p1_idx, 0], self.initial_points_xy[p1_idx, 1]
                pt2_x, pt2_y = self.initial_points_xy[p2_idx, 0], self.initial_points_xy[p2_idx, 1]
                plt.plot([pt1_x, pt2_x], [pt1_y, pt2_y], 'b-', linewidth=3)
        
        # Plot nodes only if they are within typical image bounds for clarity
        # This is a heuristic, as iniLocH/W might be intentionally out of bounds.
        valid_initial_points = []
        for i in range(self.num_targets):
            if 0 <= self.initial_points_xy[i,0] < self.imgSize[1]*1.5 and \
               0 <= self.initial_points_xy[i,1] < self.imgSize[0]*1.5: # Allow some margin
                 valid_initial_points.append(self.initial_points_xy[i,:])
        if valid_initial_points:
            valid_initial_points = np.array(valid_initial_points)
            plt.plot(valid_initial_points[:,0], valid_initial_points[:,1], 'or', markersize=6) # markersize in plt.plot
            for i in range(self.num_targets): # Label all, even if not plotted as 'or'
                 plt.text(self.initial_points_xy[i,0] + 15, self.initial_points_xy[i,1] -15 , str(i + 1), fontsize=10)
        
        plt.gca().invert_yaxis() # Match image coordinate system
        plt.title("Initial Truss Skeleton (0-based indices)")
        plt.xlabel("X-coordinate")
        plt.ylabel("Y-coordinate")
        plt.axis('equal')
        plt.show()

    def _imclearborder(self, img_bw, connectivity=8):
        """
        Removes connected components touching the border of the image.
        Args:
            img_bw (numpy.ndarray): Binary image (uint8, 0 or 255).
            connectivity (int): Not directly used in this OpenCV approach but kept for conceptual similarity.
        Returns:
            numpy.ndarray: Binary image with border components removed.
        """
        img_out = img_bw.copy()
        h, w = img_bw.shape

        # Find contours that are candidates for removal
        contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # Check if any point of the contour is on the border
            on_border = False
            for point_wrapper in contour:
                x, y = point_wrapper[0] # Contour points are wrapped in an extra array
                if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                    on_border = True
                    break
            
            if on_border:
                # Fill the contour with black (0)
                cv2.drawContours(img_out, [contour], -1, 0, thickness=cv2.FILLED)
        return img_out

    def _bwareaopen(self, img_bw, min_area):
        """
        Removes connected components smaller than min_area.
        Args:
            img_bw (numpy.ndarray): Binary image.
            min_area (int): Minimum area of components to keep.
        Returns:
            numpy.ndarray: Binary image with small components removed.
        """
        img_out = np.zeros_like(img_bw)
        contours, _ = cv2.findContours(img_bw.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) >= min_area:
                cv2.drawContours(img_out, [contour], -1, 255, thickness=cv2.FILLED)
        return img_out

    def process_images(self):
        """Main processing loop for target detection and video creation."""
        
        output_video_width, output_video_height = 1500, 600 # As per MATLAB figure size
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(self.filename_video, fourcc, self.camerafs, 
                                    (output_video_width, output_video_height))

        plt.ioff() # Turn off interactive mode for matplotlib
        self.fig_plot_combined = plt.figure(figsize=(output_video_width/100, output_video_height/100), dpi=100)

        num_cols_targets = len(self.iniLocW) # Number of targets per row (e.g., 7)

        for imgIdx in range(self.nImg):
            img_bgr = cv2.imread(self.imgPathList[imgIdx])
            if img_bgr is None:
                print(f"Warning: Could not read image {self.imgPathList[imgIdx]}")
                if imgIdx > 0:
                    self.LocPt[:, :, imgIdx] = self.LocPt[:, :, imgIdx - 1]
                else:
                    self.LocPt[:, :, imgIdx] = self.initial_points_xy
                # Create a blank frame for the video if image load fails
                blank_frame = np.zeros((output_video_height, output_video_width, 3), dtype=np.uint8)
                cv2.putText(blank_frame, f"Error loading frame {imgIdx}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
                out_video.write(blank_frame)
                continue

            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # Binarize image (MATLAB: ~(im2bw(img,threshTarget)))
            # intensity > level -> 0, else 1 (255). Equivalent to THRESH_BINARY_INV.
            _, img_bw = cv2.threshold(img_gray, self.threshTarget * 255, 255, cv2.THRESH_BINARY_INV)
            
            current_frame_loc_pt = np.zeros((self.num_targets, 2))

            for rowIdx in range(len(self.iniLocH)): # Iterate through each row of targets
                y_center_of_row = self.iniLocH[rowIdx] # 0-based y-coordinate for the center of the current row
                strip_half_h = int(np.ceil(self.targetStripH / 2.0))
                
                # Define the y-range for the search strip (0-based)
                y_start_py = int(y_center_of_row - strip_half_h)
                y_end_py   = int(y_center_of_row + strip_half_h) 

                # Clamp strip to image boundaries
                y_start_py_clamped = max(0, y_start_py)
                y_end_py_clamped   = min(self.imgSize[0], y_end_py)

                # Point indices in LocPt for the current row
                row_start_locpt_idx = rowIdx * num_cols_targets
                row_end_locpt_idx = row_start_locpt_idx + num_cols_targets

                if y_start_py_clamped >= y_end_py_clamped: # Strip is empty or invalid
                    if self.DEBUG: print(f"Warning: Invalid/empty strip for row {rowIdx} (y_center={y_center_of_row}). Using fallback.")
                    if imgIdx > 0: # Use previous frame's locations
                        current_frame_loc_pt[row_start_locpt_idx:row_end_locpt_idx, :] = \
                            self.LocPt[row_start_locpt_idx:row_end_locpt_idx, :, imgIdx - 1]
                    else: # Use initial locations
                        current_frame_loc_pt[row_start_locpt_idx:row_end_locpt_idx, :] = \
                            self.initial_points_xy[row_start_locpt_idx:row_end_locpt_idx, :]
                    continue

                img_temp_bw = img_bw[y_start_py_clamped:y_end_py_clamped, :]
                
                # Morphological operations
                img_temp_bw = self._imclearborder(img_temp_bw)
                img_temp_bw = cv2.morphologyEx(img_temp_bw, cv2.MORPH_CLOSE, self.se)
                
                # Area filtering: xor(bwareaopen(LB), bwareaopen(UB)) -> keep areas between LB and UB
                img_bwao_lb = self._bwareaopen(img_temp_bw, self.targetAreaLB)
                img_bwao_ub = self._bwareaopen(img_temp_bw, self.targetAreaUB)
                img_temp_bw_filtered = cv2.bitwise_xor(img_bwao_lb, img_bwao_ub)
                
                # Find contours (blobs) in the processed strip
                contours, _ = cv2.findContours(img_temp_bw_filtered.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                detected_blobs_stats = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    # Area check is implicitly handled by xor(bwareaopen) logic, but good to have as sanity check
                    # if not (self.targetAreaLB <= area < self.targetAreaUB): continue

                    M = cv2.moments(contour)
                    if M["m00"] == 0: continue # Avoid division by zero for centroid
                    
                    cx_strip = M["m10"] / M["m00"]
                    cy_strip = M["m01"] / M["m00"]
                    
                    # Convert centroid to full image coordinates
                    cx_full_img = cx_strip
                    cy_full_img = cy_strip + y_start_py_clamped 
                    
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter == 0: continue
                    # Circularity: MATLAB formula: perimeter^2 / (4*pi*Area). Perfect circle = 1.
                    circularity = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
                    
                    detected_blobs_stats.append({
                        'centroid': np.array([cx_full_img, cy_full_img]),
                        'area': area,
                        'circularity': circularity
                    })
                
                if len(detected_blobs_stats) < num_cols_targets:
                    # Fallback: Not enough blobs detected
                    if imgIdx > 0:
                        current_frame_loc_pt[row_start_locpt_idx:row_end_locpt_idx, :] = \
                            self.LocPt[row_start_locpt_idx:row_end_locpt_idx, :, imgIdx - 1]
                    else:
                        current_frame_loc_pt[row_start_locpt_idx:row_end_locpt_idx, :] = \
                            self.initial_points_xy[row_start_locpt_idx:row_end_locpt_idx, :]
                else:
                    # Condition 1: Filter by circularity and y-range deviation
                    cond1_filtered_stats = []
                    for stat in detected_blobs_stats:
                        # Circularity check (MATLAB: abs(circularity-1) < 1.0)
                        cond_R = abs(stat['circularity'] - 1.0) < self.circularity_deviation_threshold
                        
                        # Y-range check (deviation from this row's initial y-center)
                        yrange_deviation = abs(stat['centroid'][1] - y_center_of_row)
                        cond_C = yrange_deviation < 30 # Max 30 pixel y-deviation from initial row y

                        if cond_R and cond_C:
                            cond1_filtered_stats.append(stat)
                    
                    if len(cond1_filtered_stats) < num_cols_targets:
                        # Fallback: Not enough blobs after Condition 1
                        if imgIdx > 0:
                            current_frame_loc_pt[row_start_locpt_idx:row_end_locpt_idx, :] = \
                                self.LocPt[row_start_locpt_idx:row_end_locpt_idx, :, imgIdx - 1]
                        else:
                             current_frame_loc_pt[row_start_locpt_idx:row_end_locpt_idx, :] = \
                                self.initial_points_xy[row_start_locpt_idx:row_end_locpt_idx, :]
                    else:
                        # Condition 2: Sort by area (descending), take top N
                        cond1_filtered_stats.sort(key=lambda s: s['area'], reverse=True)
                        best_n_stats = cond1_filtered_stats[:num_cols_targets]
                        
                        # Sort these N points by x-coordinate for consistent ordering
                        best_n_stats.sort(key=lambda s: s['centroid'][0])
                        
                        # Assign to current_frame_loc_pt for this row
                        for i in range(num_cols_targets):
                            pt_idx_in_locpt = row_start_locpt_idx + i
                            current_frame_loc_pt[pt_idx_in_locpt, :] = best_n_stats[i]['centroid']

                        # Final check: Large movement condition
                        current_x_coords_row = current_frame_loc_pt[row_start_locpt_idx:row_end_locpt_idx, 0]
                        reverted_due_to_movement = False
                        if imgIdx > 0:
                            prev_x_coords_row = self.LocPt[row_start_locpt_idx:row_end_locpt_idx, 0, imgIdx - 1]
                            if np.any(np.abs(current_x_coords_row - prev_x_coords_row) > self.targetLargeMovement):
                                current_frame_loc_pt[row_start_locpt_idx:row_end_locpt_idx, :] = \
                                    self.LocPt[row_start_locpt_idx:row_end_locpt_idx, :, imgIdx - 1]
                                reverted_due_to_movement = True
                        else: # First frame, compare with initial x-coordinates for this row
                            initial_x_coords_row = self.initial_points_xy[row_start_locpt_idx:row_end_locpt_idx, 0]
                            if np.any(np.abs(current_x_coords_row - initial_x_coords_row) > self.targetLargeMovement):
                                current_frame_loc_pt[row_start_locpt_idx:row_end_locpt_idx, :] = \
                                    self.initial_points_xy[row_start_locpt_idx:row_end_locpt_idx, :]
                                reverted_due_to_movement = True
                        # if reverted_due_to_movement and self.DEBUG:
                        #     print(f"Row {rowIdx} reverted due to large movement.")
            
            self.LocPt[:, :, imgIdx] = current_frame_loc_pt

            # --- Plotting for video frame ---
            self.fig_plot_combined.clf() # Clear figure for new frame
            loc_plot_xy_current_frame = self.LocPt[:, :, imgIdx]

            # Define layout using GridSpec for more control, similar to MATLAB subplots
            gs_fig = GridSpec(4, 3, figure=self.fig_plot_combined, hspace=0.4, wspace=0.3)

            # Ax1: Test image with detected points (spans 4 rows, 1st column)
            ax1 = self.fig_plot_combined.add_subplot(gs_fig[:, 0])
            img_display_ax1 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            ax1.imshow(img_display_ax1)
            ax1.plot(loc_plot_xy_current_frame[:, 0], loc_plot_xy_current_frame[:, 1], 'o', color='lime', markersize=6, markeredgewidth=1) # Green circles
            ax1.set_title('Video', fontweight='bold', fontsize=10)
            ax1.axis('off')

            # Ax2: Stick figure (spans 4 rows, 2nd column)
            ax2 = self.fig_plot_combined.add_subplot(gs_fig[:, 1])
            # Black background for stick figure, matching original image aspect ratio potentially
            ax2.set_facecolor('black') 
            ax2.set_xlim(0, self.imgSize[1])
            ax2.set_ylim(self.imgSize[0], 0) # Inverted Y for image coordinates
            for p_map_idx in range(self.ptMap.shape[0]):
                pt1_global_idx, pt2_global_idx = self.ptMap[p_map_idx, 0], self.ptMap[p_map_idx, 1]
                pt1_coord = loc_plot_xy_current_frame[pt1_global_idx, :]
                pt2_coord = loc_plot_xy_current_frame[pt2_global_idx, :]
                ax2.plot([pt1_coord[0], pt2_coord[0]], [pt1_coord[1], pt2_coord[1]], 
                         color='blue', linewidth=2)
            ax2.plot(loc_plot_xy_current_frame[:, 0], loc_plot_xy_current_frame[:, 1], 'o', color='lime', markersize=6, markeredgewidth=1)# Green circles
            ax2.set_title('Skeleton Figure', fontweight='bold', fontsize=10)
            ax2.axis('off')

            # Ax3: X displacement at "left column" (MATLAB points 22-28 -> 0-based 21-27, last row)
            ax3 = self.fig_plot_combined.add_subplot(gs_fig[0:2, 2]) # Top two rows of 3rd column
            # Points 21 to 27 (0-based) are the last 7 points (bottom row in a 4x7 grid)
            pts_indices_bottom_row = np.arange(self.num_targets - num_cols_targets, self.num_targets)
            if imgIdx >= 0 : # Plot even for the first frame (displacement will be 0)
                xdisp_bottom_row = self.LocPt[pts_indices_bottom_row, 0, :imgIdx+1]
                # Displacement relative to the first frame's position for each point
                xdisp_relative_br = xdisp_bottom_row - xdisp_bottom_row[:, 0][:, np.newaxis]
                time_axis = np.arange(imgIdx + 1) / self.camerafs
                for i in range(xdisp_relative_br.shape[0]):
                    ax3.plot(time_axis, xdisp_relative_br[i, :], linewidth=1.5)
            ax3.set_ylabel('X disp. bottom row (px)', fontsize=8, fontweight='bold')
            ax3.tick_params(axis='both', which='major', labelsize=8)
            ax3.grid(True)

            # Ax4: X displacement at "top floor" (MATLAB points 1, 8, 15, 22 -> 0-based 0, 7, 14, 21, first point of each row)
            ax4 = self.fig_plot_combined.add_subplot(gs_fig[2:4, 2]) # Bottom two rows of 3rd column
            pts_indices_first_col = np.arange(0, self.num_targets, num_cols_targets) # 0, 7, 14, 21
            if imgIdx >= 0:
                xdisp_first_col = self.LocPt[pts_indices_first_col, 0, :imgIdx+1]
                xdisp_relative_fc = xdisp_first_col - xdisp_first_col[:, 0][:, np.newaxis]
                for i in range(xdisp_relative_fc.shape[0]):
                    ax4.plot(time_axis, xdisp_relative_fc[i, :], linewidth=1.5)
            ax4.set_xlabel('Time(s)', fontsize=8, fontweight='bold')
            ax4.set_ylabel('X disp. first col (px)', fontsize=8, fontweight='bold')
            ax4.tick_params(axis='both', which='major', labelsize=8)
            ax4.grid(True)
            
            # Render matplotlib figure to numpy array
            self.fig_plot_combined.canvas.draw()
            buf = self.fig_plot_combined.canvas.buffer_rgba()
            # 将缓冲区转换为 numpy 数组
            plot_img_array_rgba = np.frombuffer(buf, dtype=np.uint8)
            # 将数组重塑为正确的维度（高度、宽度、4 个通道用于 RGBA）
            # 注意：get_width_height() 返回 (宽度, 高度)，因此 [::-1] 使其变为 (高度, 宽度)
            canvas_height, canvas_width = self.fig_plot_combined.canvas.get_width_height()[::-1]
            plot_img_array_rgba = plot_img_array_rgba.reshape(canvas_height, canvas_width, 4)
            # 将 RGBA 转换为 RGB
            plot_img_array = cv2.cvtColor(plot_img_array_rgba, cv2.COLOR_RGBA2RGB)
            # 下一行将 RGB 转换为 BGR，保持不变
            plot_img_bgr = cv2.cvtColor(plot_img_array, cv2.COLOR_RGB2BGR)
            
            # Resize if necessary to fit video dimensions
            if plot_img_bgr.shape[1] != output_video_width or plot_img_bgr.shape[0] != output_video_height:
                 plot_img_bgr = cv2.resize(plot_img_bgr, (output_video_width, output_video_height))

            out_video.write(plot_img_bgr)
            
            print(f'Processing ({imgIdx + 1} / {self.nImg})')

        plt.close(self.fig_plot_combined)
        out_video.release()
        print('Python script complete!! Video saved to:', self.filename_video)

    def run(self):
        """Executes the image processing workflow."""
        self.process_images()

if __name__ == '__main__':
    # --- Configuration for running ---
    # IMPORTANT: Replace 'img_folder_path' with the actual path to your image directory.
    img_folder_path = 'img'  # Default: 'img' subdirectory relative to this script
    output_main_folder = 'result_py' # Default: 'result_py' subdirectory

    # --- (Optional) Create dummy images for testing if 'img' folder is empty or doesn't exist ---
    create_dummy_images = False # Set to True to generate dummy images
    if not os.path.exists(img_folder_path) or not glob.glob(os.path.join(img_folder_path, '*.*')):
        print(f"Image folder '{img_folder_path}' not found or empty.")
        create_dummy_images = True # Force dummy image creation if folder is problematic

    if create_dummy_images:
        print(f"Attempting to create dummy images in '{img_folder_path}'...")
        if not os.path.exists(img_folder_path):
            os.makedirs(img_folder_path)
        
        dummy_img_height, dummy_img_width = 1300, 2046 # Match MATLAB's example imgSize
        num_dummy_frames = 10

        # Define plausible iniLocH/W for these dummy images (0-based)
        # These should be within the dummy_img_height and dummy_img_width
        dummy_iniLocH = np.array([200, 400, 600, 800]) # Example y-coordinates
        dummy_iniLocW = np.array([300, 500, 700, 900, 1100, 1300, 1500]) # Example x-coordinates
        
        num_rows_dummy, num_cols_dummy = len(dummy_iniLocH), len(dummy_iniLocW)

        for i in range(num_dummy_frames):
            dummy_img = np.zeros((dummy_img_height, dummy_img_width, 3), dtype=np.uint8)
            for r_idx in range(num_rows_dummy):
                for c_idx in range(num_cols_dummy):
                    # Simulate some horizontal sinusoidal movement for targets
                    x_center = dummy_iniLocW[c_idx] + int(20 * np.sin(2 * np.pi * i / num_dummy_frames + c_idx * 0.5))
                    y_center = dummy_iniLocH[r_idx]
                    cv2.circle(dummy_img, (x_center, y_center), 10, (255,255,255), -1) # White circles
            cv2.imwrite(os.path.join(img_folder_path, f'dummy_frame_{i:03d}.jpg'), dummy_img)
        print(f"Created {num_dummy_frames} dummy images in '{img_folder_path}'.")
        print("IMPORTANT: If using these dummy images, the VisionVibrationMeasurement class's")
        print("iniLocH and iniLocW parameters might need to be adjusted to match dummy_iniLocH/W.")
        # --- End of dummy image creation ---

    try:
        analyzer = VisionVibrationMeasurement(img_dir=img_folder_path, output_folder=output_main_folder)
        
        # If you created dummy images and want to use their specific iniLocH/W:
        if create_dummy_images:
             print("Adjusting iniLocH/W for dummy images.")
             analyzer.iniLocH = dummy_iniLocH 
             analyzer.iniLocW = dummy_iniLocW
             analyzer.num_targets = len(analyzer.iniLocH) * len(analyzer.iniLocW)
             analyzer.LocPt = np.zeros((analyzer.num_targets, 2, analyzer.nImg)) # Reinitialize
             # Recompute initial_points_xy based on new iniLocH/W
             pts_x_init_dummy = np.tile(analyzer.iniLocW, len(analyzer.iniLocH))
             pts_y_init_dummy = np.repeat(analyzer.iniLocH, len(analyzer.iniLocW))
             analyzer.initial_points_xy = np.zeros((analyzer.num_targets, 2))
             for i in range(analyzer.num_targets):
                 analyzer.initial_points_xy[i, 0] = pts_x_init_dummy[i]
                 analyzer.initial_points_xy[i, 1] = pts_y_init_dummy[i]
             # Also need to re-calculate ptMap if num_targets changed, but it should be 4x7=28
             # For simplicity, assume dummy targets also form a 4x7 grid for ptMap to remain valid.

        analyzer.run()
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the image directory is correctly specified and contains images.")
    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
