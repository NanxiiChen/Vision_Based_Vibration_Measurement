import cv2
import numpy as np
import os
import glob
import math
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from skimage import measure, morphology, segmentation

class VisionBasedVibrationMeasurement:
    def __init__(self, config):
        """
        Initializes the vision-based vibration measurement system.
        Args:
            config (dict): A dictionary containing configuration parameters.
        """
        self.folder_out = config.get('folder_out', 'result_py')
        self.debug = config.get('debug', False)
        
        self.img_path_list = []
        self.n_img = 0
        # MATLAB imgSize = [rows, cols] -> height, width
        self.img_size = tuple(config.get('img_size', (1300, 2046))) 
        
        self._setup_paths(config.get('img_folder', 'img'))
        self._initialize_parameters(config)

        if not os.path.exists(self.folder_out):
            os.makedirs(self.folder_out)
        
        self.loc_pt = None 
        self.output_video_writer = None

        if self.debug and self.n_img > 0:
            self._plot_initial_targets_debug()
            self._plot_initial_skeleton_debug()

    def _setup_paths(self, img_folder_name):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_folder_abs = os.path.join(script_dir, img_folder_name)
        
        search_pattern = os.path.join(img_folder_abs, '*.jpg')
        raw_paths = glob.glob(search_pattern)
        self.img_path_list = sorted(raw_paths) # Sort for consistent order
        self.n_img = len(self.img_path_list)
        
        if self.n_img == 0:
            raise FileNotFoundError(f"No JPG images found in {search_pattern}. Searched in absolute path: {img_folder_abs}")

    def _initialize_parameters(self, config):
        self.ini_loc_h = np.array(config.get('ini_loc_h', [1904, 1420, 931, 471]))
        self.ini_loc_w = np.array(config.get('ini_loc_w', [148, 310, 466, 625, 778, 940, 1111]))
        self.target_strip_h = config.get('target_strip_h', 100)
        self.target_size = config.get('target_size', 20) # Used for debug plot radius

        # Point connection map (0-indexed for Python)
        pt_h_data = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7],
                     [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14],
                     [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21],
                     [22, 23], [23, 24], [24, 25], [25, 26], [26, 27], [27, 28]]
        pt_h = np.array(pt_h_data) - 1

        pt_v_tmp_data = [[1, 8], [8, 15], [15, 22]]
        pt_v_tmp = np.array(pt_v_tmp_data) - 1 
        
        pt_v_list = []
        for offset in [0, 3, 6]: # For 1st, 4th, and 7th columns of points
            pt_v_list.append(pt_v_tmp + offset)
        pt_v = np.vstack(pt_v_list)
        self.pt_map = np.vstack((pt_h, pt_v))

        self.thresh_target = config.get('thresh_target', 0.1) # For im2bw
        self.target_area_lb = config.get('target_area_lb', 100)
        self.target_area_ub = config.get('target_area_ub', 500)
        self.target_large_movement = config.get('target_large_movement', 100)
        # MATLAB: abs(circularity-1) < 1. This '1' is the deviation threshold.
        self.circularity_deviation_threshold = config.get('circularity_deviation_threshold', 1.0) 

        disk_radius = 3
        self.se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, 
                                            (2 * disk_radius + 1, 2 * disk_radius + 1))
        self.camera_fs = config.get('camera_fs', 20)
        video_filename = config.get('video_filename_stem', 'vibration_py_output') + '.mp4'
        self.filename_video = os.path.join(self.folder_out, video_filename)

    def _plot_initial_targets_debug(self):
        img_tmp_bgr = cv2.imread(self.img_path_list[0])
        if img_tmp_bgr is None: return
        img_tmp_rgb = cv2.cvtColor(img_tmp_bgr, cv2.COLOR_BGR2RGB)

        circle_loc_x = np.tile(self.ini_loc_w, 4)
        circle_loc_y = np.repeat(self.ini_loc_h, len(self.ini_loc_w)) # 7 points per row
        
        plt.figure(figsize=(10,7))
        plt.imshow(img_tmp_rgb)
        plt.title("Initial Target Locations (Debug)")
        for i in range(len(circle_loc_x)):
            circ = plt.Circle((circle_loc_x[i], circle_loc_y[i]), self.target_size / 2,
                              color='red', fill=False, linewidth=3)
            plt.gca().add_patch(circ)
        plt.savefig(os.path.join(self.folder_out, "initial_targets_debug.png"))
        plt.close()

    def _plot_initial_skeleton_debug(self):
        pt_x = np.tile(self.ini_loc_w, 4)
        pt_y = np.repeat(self.ini_loc_h, len(self.ini_loc_w))

        plt.figure(figsize=(10,7))
        plt.title("Initial Truss Skeleton (Debug)")
        for i in range(self.pt_map.shape[0]):
            idx1, idx2 = self.pt_map[i, 0], self.pt_map[i, 1]
            plt.plot([pt_x[idx1], pt_x[idx2]], [pt_y[idx1], pt_y[idx2]],
                     color='blue', linewidth=3)
        plt.plot(pt_x, pt_y, 'or', markersize=6) # 'or' means red circles
        for i in range(len(pt_x)):
            plt.text(pt_x[i] + 15, pt_y[i] + 15, str(i + 1), fontsize=10, color='white')
        
        plt.gca().invert_yaxis() 
        plt.gca().set_facecolor('lightgray')
        plt.savefig(os.path.join(self.folder_out, "initial_skeleton_debug.png"))
        plt.close()

    def _get_frame_from_figure(self, fig):
        fig.canvas.draw()
        
        # 使用 print_to_buffer() 获取 RGBA 缓冲区
        # 它返回一个元组 (buffer, (width, height))
        buf, size = fig.canvas.print_to_buffer()
        width, height = size
        
        # 将缓冲区转换为 NumPy 数组，形状为 (height, width, 4)
        img_rgba = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 4))
        
        # 提取 RGB 分量 (丢弃 alpha 通道)
        # 结果数组形状为 (height, width, 3)
        img_rgb = img_rgba[:, :, :3]
        
        # 将 RGB 转换为 BGR 以便 OpenCV 使用
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        return img_bgr
    def process_video(self):
        if self.n_img == 0:
            print("No images to process.")
            return

        self.loc_pt = np.zeros((28, 2, self.n_img)) # 28 points, (x,y), n_img frames

        fig_width_px, fig_height_px = 1500, 600 # Match MATLAB's figure size for video
        fig_dpi = 100
        fig_width_inches = fig_width_px / fig_dpi
        fig_height_inches = fig_height_px / fig_dpi

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video_writer = cv2.VideoWriter(self.filename_video, fourcc, self.camera_fs, 
                                                   (fig_width_px, fig_height_px))

        fig = plt.figure(figsize=(fig_width_inches, fig_height_inches), dpi=fig_dpi)

        # Update actual image size from the first loaded image
        first_img_bgr = cv2.imread(self.img_path_list[0])
        if first_img_bgr is None:
            print(f"Error: Could not load the first image: {self.img_path_list[0]}")
            self.output_video_writer.release()
            plt.close(fig)
            return
        self.img_size = (first_img_bgr.shape[0], first_img_bgr.shape[1]) # height, width

        for img_idx in range(self.n_img):
            print(f"Processing ({img_idx + 1} / {self.n_img})")

            img_bgr = cv2.imread(self.img_path_list[img_idx])
            if img_bgr is None:
                print(f"Warning: Could not read image {self.img_path_list[img_idx]}. Using previous frame's points.")
                if img_idx > 0:
                    self.loc_pt[:, :, img_idx] = self.loc_pt[:, :, img_idx - 1]
                else: # First image failed, use initial positions
                    pt_x_initial = np.tile(self.ini_loc_w, 4)
                    pt_y_initial = np.repeat(self.ini_loc_h, len(self.ini_loc_w))
                    self.loc_pt[:, 0, img_idx] = pt_x_initial
                    self.loc_pt[:, 1, img_idx] = pt_y_initial
                # Create a blank frame or skip? For now, we'll plot with these points.
                # continue # Or try to plot with these points

            current_img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
            # MATLAB: imgBW = ~(im2bw(img,threshTarget));
            # im2bw: val > thresh -> 1, else 0. Then invert.
            # cv2.THRESH_BINARY_INV: val > thresh -> 0, else maxval
            _, current_img_bw = cv2.threshold(current_img_gray, 
                                              self.thresh_target * 255, 255, 
                                              cv2.THRESH_BINARY_INV)
            current_img_bw = current_img_bw.astype(np.uint8)

            for row_idx in range(len(self.ini_loc_h)): # 0 to 3
                # Define vertical strip for target search
                th_center_y = self.ini_loc_h[row_idx]
                th_half_h = math.ceil(self.target_strip_h / 2.0)
                
                # MATLAB: targetH = iniLocH(rowIdx)-ceil(targetStripH/2)+1 : iniLocH(rowIdx)+ceil(targetStripH/2);
                # 1-based indices for start and end of strip in MATLAB
                y_start_1based = th_center_y - th_half_h + 1
                y_end_1based = th_center_y + th_half_h

                # Clamp to image boundaries (1-based)
                y_start_1based = max(1, y_start_1based)
                y_end_1based = min(self.img_size[0], y_end_1based) # img_size[0] is height

                # Convert to 0-based Python slice indices
                y_slice_start = int(round(y_start_1based - 1))
                y_slice_end = int(round(y_end_1based)) # Exclusive end for Python slice

                if y_slice_start >= y_slice_end:
                    print(f"Warning: Empty target strip for row {row_idx} at image {img_idx}.")
                    # Fallback for this row
                    start_pt_idx = row_idx * len(self.ini_loc_w)
                    end_pt_idx = (row_idx + 1) * len(self.ini_loc_w)
                    if img_idx > 0:
                        self.loc_pt[start_pt_idx:end_pt_idx, :, img_idx] = self.loc_pt[start_pt_idx:end_pt_idx, :, img_idx - 1]
                    else:
                        self.loc_pt[start_pt_idx:end_pt_idx, 0, img_idx] = self.ini_loc_w
                        self.loc_pt[start_pt_idx:end_pt_idx, 1, img_idx] = np.full(len(self.ini_loc_w), self.ini_loc_h[row_idx])
                    continue
                
                img_temp_bw = current_img_bw[y_slice_start:y_slice_end, :]

                # Morphological operations
                img_temp_bw = segmentation.clear_border(img_temp_bw > 0).astype(np.uint8) * 255
                img_temp_bw = cv2.morphologyEx(img_temp_bw, cv2.MORPH_CLOSE, self.se)

                # Filter by area: targetAreaLB <= area < targetAreaUB
                contours, _ = cv2.findContours(img_temp_bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                img_temp_bw_filtered = np.zeros_like(img_temp_bw)
                valid_contours_for_props = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if self.target_area_lb <= area < self.target_area_ub:
                        valid_contours_for_props.append(contour)
                cv2.drawContours(img_temp_bw_filtered, valid_contours_for_props, -1, 255, thickness=cv2.FILLED)
                img_temp_bw = img_temp_bw_filtered
                
                # Region properties
                label_image = measure.label(img_temp_bw > 0, connectivity=2) # 8-connectivity
                props = measure.regionprops(label_image)

                centers_coords_list = []
                areas_list = []
                circularity_list = []

                if props:
                    for p_idx, p in enumerate(props):
                        # Centroid from skimage is (row, col) -> (y, x) relative to img_temp_bw
                        # Convert to (x, y) global coordinates
                        center_x_global = p.centroid[1] 
                        center_y_global = p.centroid[0] + y_slice_start
                        centers_coords_list.append([center_x_global, center_y_global])
                        
                        areas_list.append(p.area)
                        
                        # Circularity P^2 / (4*pi*A)
                        perimeter = p.perimeter
                        if p.area > 0 and perimeter > 0: # Avoid division by zero
                            circ = (perimeter ** 2) / (4 * np.pi * p.area)
                            circularity_list.append(circ)
                        else:
                            circularity_list.append(1000) # A large number if area or perimeter is zero
                
                centers_found_np = np.array(centers_coords_list)
                areas_np = np.array(areas_list)
                circularity_np = np.array(circularity_list)

                # y-range relative to initial row height
                if centers_found_np.shape[0] > 0:
                    yrange = np.round(centers_found_np[:, 1] - self.ini_loc_h[row_idx])
                else:
                    yrange = np.empty((0,))

                current_row_pt_start_idx = row_idx * len(self.ini_loc_w)
                current_row_pt_end_idx = (row_idx + 1) * len(self.ini_loc_w)
                num_targets_per_row = len(self.ini_loc_w)

                if centers_found_np.shape[0] < num_targets_per_row:
                    use_previous = True
                else:
                    # Condition 1: Circularity and y-range
                    ind_r = np.abs(circularity_np - 1.0) < self.circularity_deviation_threshold
                    ind_c = np.abs(yrange) < 30 # y-deviation limit
                    
                    indices_cond1 = np.where(np.logical_and(ind_r, ind_c))[0]

                    if len(indices_cond1) < num_targets_per_row:
                        use_previous = True
                    else:
                        # Condition 2: Sort by area (descending) among those satisfying cond1
                        areas_cond1 = areas_np[indices_cond1]
                        sorted_indices_within_cond1 = np.argsort(-areas_cond1) # Sorts indices of areas_cond1
                        
                        # Get global indices of best candidates
                        best_global_indices = indices_cond1[sorted_indices_within_cond1]
                        
                        centers_filtered = centers_found_np[best_global_indices, :]
                        
                        # Pick top N (num_targets_per_row)
                        if centers_filtered.shape[0] >= num_targets_per_row:
                            centers_top_n = centers_filtered[:num_targets_per_row, :]
                            
                            # Sort these N points by x-coordinate
                            sort_by_x_indices = np.argsort(centers_top_n[:, 0])
                            final_centers_for_row = centers_top_n[sort_by_x_indices, :]
                            
                            self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, :, img_idx] = final_centers_for_row
                            use_previous = False # Successfully found points

                            # Last check: Large movement
                            if img_idx > 0:
                                prev_x = self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, 0, img_idx - 1]
                                curr_x = self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, 0, img_idx]
                                if np.any(np.abs(curr_x - prev_x) > self.target_large_movement):
                                    use_previous = True 
                            else: # First image
                                initial_x = self.ini_loc_w
                                curr_x = self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, 0, img_idx]
                                if np.any(np.abs(curr_x - initial_x) > self.target_large_movement):
                                    # Use initial if large deviation from expected start
                                    self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, 0, img_idx] = self.ini_loc_w
                                    self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, 1, img_idx] = np.full(num_targets_per_row, self.ini_loc_h[row_idx])
                                    use_previous = False # Already handled by setting to initial
                        else: # Not enough points after area sort
                            use_previous = True
                
                if use_previous:
                    if img_idx > 0:
                        self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, :, img_idx] = \
                            self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, :, img_idx - 1]
                    else:
                        self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, 0, img_idx] = self.ini_loc_w
                        self.loc_pt[current_row_pt_start_idx:current_row_pt_end_idx, 1, img_idx] = \
                            np.full(num_targets_per_row, self.ini_loc_h[row_idx])
            # End row_idx loop

            # Plotting for video frame
            fig.clf() 
            loc_plot_x = self.loc_pt[:, 0, img_idx]
            loc_plot_y = self.loc_pt[:, 1, img_idx]

            # Subplot 1: Video frame with markers
            ax1 = plt.subplot2grid((4,3), (0,0), rowspan=4, fig=fig)
            ax1.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            ax1.plot(loc_plot_x, loc_plot_y, 'o', color="lime", markersize=4, markeredgewidth=1)
            ax1.set_xlabel('Video', fontweight='bold', fontsize=10)
            ax1.axis('off')

            # Subplot 2: Skeleton figure
            ax2 = plt.subplot2grid((4,3), (0,1), rowspan=4, fig=fig)
            ax2.set_facecolor('black')
            ax2.set_xlim(0, self.img_size[1]) # width
            ax2.set_ylim(self.img_size[0], 0) # height, inverted y
            for i in range(self.pt_map.shape[0]):
                idx1, idx2 = self.pt_map[i, 0], self.pt_map[i, 1]
                ax2.plot([loc_plot_x[idx1], loc_plot_x[idx2]],
                         [loc_plot_y[idx1], loc_plot_y[idx2]],
                         color='blue', linewidth=2)
            ax2.plot(loc_plot_x, loc_plot_y, 'o', color="lime", markersize=5, markeredgewidth=1)
            ax2.set_xlabel('Skeleton figure', fontweight='bold', fontsize=10)
            ax2.axis('off')

            # Subplot 3: X displacement at "bottom row" (MATLAB points 22-28)
            ax3 = plt.subplot2grid((4,3), (0,2), rowspan=2, fig=fig)
            if img_idx >= 0:
                # MATLAB points 22-28 are indices 21-27 (last 7 points, 4th row)
                points_idx_ax3 = np.arange(21, 28) 
                x_disp_all_frames_ax3 = self.loc_pt[points_idx_ax3, 0, :img_idx + 1]
                if x_disp_all_frames_ax3.shape[1] > 0:
                    initial_x_ax3 = x_disp_all_frames_ax3[:, 0].reshape(-1, 1)
                    x_disp_relative_ax3 = x_disp_all_frames_ax3 - initial_x_ax3
                    time_axis = (np.arange(img_idx + 1)) / self.camera_fs
                    for i in range(x_disp_relative_ax3.shape[0]):
                        ax3.plot(time_axis, x_disp_relative_ax3[i, :], linewidth=1.5)
            ax3.set_ylabel('X disp. bottom row (px)', fontweight='bold', fontsize=9)
            ax3.tick_params(labelsize=8)
            ax3.grid(True)

            # Subplot 4: X displacement at "left column" (MATLAB points 1, 8, 15, 22)
            ax4 = plt.subplot2grid((4,3), (2,2), rowspan=2, fig=fig)
            if img_idx >= 0:
                # MATLAB points 1, 8, 15, 22 are indices 0, 7, 14, 21 (first point of each row)
                points_idx_ax4 = np.array([0, 7, 14, 21]) 
                x_disp_all_frames_ax4 = self.loc_pt[points_idx_ax4, 0, :img_idx + 1]
                if x_disp_all_frames_ax4.shape[1] > 0:
                    initial_x_ax4 = x_disp_all_frames_ax4[:, 0].reshape(-1, 1)
                    x_disp_relative_ax4 = x_disp_all_frames_ax4 - initial_x_ax4
                    time_axis = (np.arange(img_idx + 1)) / self.camera_fs
                    for i in range(x_disp_relative_ax4.shape[0]):
                        ax4.plot(time_axis, x_disp_relative_ax4[i, :], linewidth=1.5)
            ax4.set_xlabel('Time(s)', fontweight='bold', fontsize=9)
            ax4.set_ylabel('X disp. left column (px)', fontweight='bold', fontsize=9)
            ax4.tick_params(labelsize=8)
            ax4.grid(True)
            
            plt.tight_layout(pad=0.5) # Adjust spacing
            frame_bgr = self._get_frame_from_figure(fig)
            self.output_video_writer.write(frame_bgr)
        # End img_idx loop

        plt.close(fig)
        self.output_video_writer.release()
        print('Complete!! Output video saved to:', self.filename_video)


if __name__ == '__main__':
    # --- Configuration ---
    # Ensure 'img' folder with .jpg images exists in the same directory as this script.
    # Example: ./img/frame001.jpg, ./img/frame002.jpg, ...
    
    # Create a dummy 'img' folder and some images if they don't exist for testing
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    dummy_img_folder = os.path.join(current_script_dir, 'img')
    if not os.path.exists(dummy_img_folder):
        os.makedirs(dummy_img_folder)
        print(f"Created dummy 'img' folder at {dummy_img_folder}")
        # Create a few dummy images
        for i in range(5): # Create 5 dummy frames
            dummy_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            # Add some dummy circles that might be detected
            cv2.circle(dummy_frame, (300 + i*10, 300), 10, (0,0,0), -1) # Dark circle
            cv2.circle(dummy_frame, (600 + i*5, 500), 10, (50,50,50), -1) # Dark circle
            cv2.imwrite(os.path.join(dummy_img_folder, f'frame{i:03d}.jpg'), dummy_frame)
        print(f"Created 5 dummy .jpg images in {dummy_img_folder}. "
              "Replace these with your actual images and adjust 'ini_loc_h/w' etc.")
        # For dummy images, the default ini_loc_h/w will likely not match.
        # The default img_size is also different from dummy images.
        # User should adjust config for their specific video/images.
        default_img_size_for_dummy = [720, 1280] # height, width for dummy
        default_ini_loc_h_dummy = [300, 500] # Approximate y-centers of dummy circles
        default_ini_loc_w_dummy = [300, 310, 320, 330, 340, 350, 360] # Dummy x-centers
    else:
        default_img_size_for_dummy = [1300, 2046] # Original MATLAB default
        default_ini_loc_h_dummy = [1904, 1420, 931, 471]
        default_ini_loc_w_dummy = [148, 310, 466, 625, 778, 940, 1111]


    config = {
        'img_folder': 'img', # Relative to script location
        'folder_out': 'result_py', # Relative to script location
        'debug': False, # Set to True to see initial plots saved as PNGs
        'img_size': default_img_size_for_dummy, # [height, width]
        'ini_loc_h': default_ini_loc_h_dummy, # Heights of rows of targets
        'ini_loc_w': default_ini_loc_w_dummy, # Widths of columns of targets
        'target_strip_h': 100,
        'target_size': 20,
        'thresh_target': 0.2, # Adjusted for potentially lighter dummy targets
        'target_area_lb': 50,  # Adjusted for potentially smaller dummy targets
        'target_area_ub': 800, # Adjusted
        'target_large_movement': 150, # Increased for dummy movement
        'circularity_deviation_threshold': 1.5, # Looser for dummy shapes
        'camera_fs': 20, # Frame rate for dummy video
        'video_filename_stem': 'vibration_py_output'
    }

    try:
        analyzer = VisionBasedVibrationMeasurement(config)
        analyzer.process_video()
    except FileNotFoundError as e:
        print(f"File Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()