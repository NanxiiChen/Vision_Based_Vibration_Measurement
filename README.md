# 基于视觉的振动测量 Python 实现

本项目使用 Python 及 OpenCV、Scikit-image 等库，通过分析图像序列来追踪视频中标记点的运动，从而实现基于视觉的振动测量。

## 项目来源与贡献

本项目是从 [chulminy/Vision_Based_Vibration_Measurement](https://github.com/chulminy/Vision_Based_Vibration_Measurement/tree/master) fork 而来。原项目提供了一个 MATLAB 程序 (`main.m`) 以及相关的示例数据。本项目的目标是将原始的 MATLAB 实现重写为 Python 版本，以便于更广泛的使用和集成。

主要的 Python 版本重写工作由 Gemini 2.5 Pro 和 NanxiChen 共同完成。

## 算法流程

程序的主要处理流程封装在 `VisionBasedVibrationMeasurement` 类中，可以分为以下几个关键步骤：

### 1. 初始化与配置

*   **加载配置**: 从传入的 `config` 字典中读取各项参数，例如：
    *   图像文件夹路径 (`img_folder`) 和结果输出文件夹路径 (`folder_out`)。
    *   `debug` 模式标志，用于控制是否输出中间调试图像。
    *   图像的预期尺寸 (`img_size`)。
    *   目标点的初始位置：
        *   `ini_loc_h`: 每行目标点的初始垂直 (Y) 坐标。
        *   `ini_loc_w`: 每列目标点的初始水平 (X) 坐标。
    *   `target_strip_h`: 用于目标检测的垂直搜索条带的高度。
    *   `target_size`: 目标点的大致尺寸（主要用于调试绘图）。
    *   图像处理相关的阈值：
        *   `thresh_target`: 图像二值化的阈值。
        *   `target_area_lb`, `target_area_ub`: 目标区域有效面积的下限和上限。
        *   `target_large_movement`: 判断目标点是否发生过大（可能错误）移动的阈值。
        *   `circularity_deviation_threshold`: 判断检测区域圆度的允许偏差。
    *   形态学操作的结构元素 (`se`)。
    *   视频的帧率 (`camera_fs`) 和输出视频文件名。
*   **路径设置 (`_setup_paths` 方法)**:
    *   根据提供的图像文件夹名称，构建绝对路径。
    *   使用 `glob` 查找所有 `.jpg` 图像文件，并按名称排序，存储图像路径列表 `img_path_list` 和图像总数 `n_img`。
    *   如果找不到图像，则抛出 `FileNotFoundError`。
*   **参数初始化 (`_initialize_parameters` 方法)**:
    *   将配置中的列表参数转换为 NumPy 数组。
    *   **动态生成骨架连接图 (`pt_map`)**: 根据 `ini_loc_h` (行数) 和 `ini_loc_w` (列数) 的实际长度，动态计算并生成一个点连接图。该图定义了哪些点之间应该绘制连线以形成骨架。这与早期版本中硬编码的连接方式不同，使其能适应不同网格大小的目标点。
    *   创建形态学闭运算所需的结构元素。
*   **创建输出文件夹**: 如果配置的输出文件夹不存在，则创建它。
*   **调试绘图 (可选)**: 如果 `debug` 模式开启且存在图像：
    *   `_plot_initial_targets_debug`: 读取第一张图像，并在其上根据 `ini_loc_h` 和 `ini_loc_w` 绘制初始目标点的红色圆圈，保存为 "initial\_targets\_debug.png"。
    *   `_plot_initial_skeleton_debug`: 根据 `ini_loc_h`、`ini_loc_w` 和动态生成的 `pt_map` 绘制初始的桁架骨架图，保存为 "initial\_skeleton\_debug.png"。

### 2. 视频处理 (`process_video` 方法)

这是程序的核心处理循环，用于分析每一帧图像并生成结果视频。

*   **初始化**:
    *   `loc_pt`: 创建一个三维 NumPy 数组，用于存储在每一帧中检测到的所有目标点的 (x, y) 坐标。其维度根据目标点的总数（行数 \* 列数）和图像总数动态确定。
    *   设置视频输出参数：定义输出视频的尺寸（与 MATLAB 版本一致，1500x600 像素）、DPI。
    *   创建 `cv2.VideoWriter` 对象，用于将处理后的帧写入 MP4 视频文件。
    *   创建 Matplotlib 的 `Figure` 对象，用于在后台渲染每一帧视频的合成图像。
    *   读取第一张图像以获取实际的图像尺寸，并更新 `self.img_size`。如果第一张图加载失败，则打印错误并退出。

*   **逐帧处理循环**: 遍历 `img_path_list` 中的每一张图像。
    *   **打印进度**: 输出当前正在处理的帧号和总帧数。
    *   **图像加载与预处理**:
        *   使用 `cv2.imread` 读取当前帧图像。
        *   如果图像加载失败（例如文件损坏或不存在），则打印警告。此时：
            *   如果不是第一帧，则直接使用前一帧检测到的目标点位置。
            *   如果是第一帧加载失败，则使用配置中定义的初始目标点位置 (`ini_loc_w`, `ini_loc_h`)。
        *   将加载的 BGR 图像转换为灰度图像 (`current_img_gray`)。
        *   **图像二值化**: 使用 `cv2.threshold` 对灰度图像进行二值化处理，并反转颜色（`cv2.THRESH_BINARY_INV`），使得感兴趣的目标点区域变为白色（值为255），背景变为黑色（值为0）。阈值来源于 `self.thresh_target`。

    *   **按行检测目标点 (内层循环)**: 假设目标点主要在水平方向运动，代码逐行（根据 `self.ini_loc_h` 定义的初始行）处理目标点。
        *   **确定垂直搜索区域 (`target_h`)**: 根据当前行的初始 Y 坐标 (`self.ini_loc_h[row_idx]`) 和条带高度 (`self.target_strip_h`)，计算出一个垂直方向的搜索范围。此范围会被限制在图像的有效边界内。
        *   **提取行条带**: 从二值化图像 `current_img_bw` 中提取出当前行对应的水平条带图像 `img_temp_bw`。
        *   **图像清理与形态学操作**:
            *   `segmentation.clear_border`: 移除接触到条带图像边界的白色区域（潜在目标点）。
            *   `cv2.morphologyEx` (闭运算): 使用之前定义的结构元素 `self.se` 对条带图像进行形态学闭运算，目的是填充目标点内部可能存在的小孔洞，并平滑其边缘。
            *   **面积过滤**:
                *   使用 `cv2.findContours` 找到闭运算后图像中的所有轮廓。
                *   遍历轮廓，计算每个轮廓的面积 (`cv2.contourArea`)。
                *   只保留面积在 `self.target_area_lb` 和 `self.target_area_ub` 之间的轮廓，过滤掉过小或过大的区域。
                *   将符合面积条件的轮廓重新绘制到一个新的空白图像上，得到 `img_temp_bw`。
        *   **区域属性分析**:
            *   `measure.label`: 对处理后的条带图像 `img_temp_bw` 进行连通区域标记。
            *   `measure.regionprops`: 计算每个标记区域的属性，主要包括：
                *   质心 (`centroid`): 提取质心的 (y, x) 坐标（相对于条带图像），并转换为完整图像坐标系下的 (x, y) 坐标。
                *   面积 (`area`)。
                *   周长 (`perimeter`)。
            *   **计算圆度**: 根据公式 `P^2 / (4 * pi * A)` 计算每个区域的圆度，其中 P 是周长，A 是面积。理想圆的圆度为1。
            *   **计算 Y 坐标偏差 (`yrange`)**: 计算检测到的每个目标点质心的 Y 坐标相对于其所在行的初始 Y 坐标 (`self.ini_loc_h[row_idx]`) 的偏差。

        *   **筛选并确定目标点位置**:
            *   获取当前行在 `self.loc_pt` 中存储的起始和结束索引，以及每行应有的目标点数量 (`num_targets_per_row`，即列数 `self.num_cols`)。
            *   **候选点数量判断**:
                *   如果通过区域属性分析找到的候选点数量少于 `num_targets_per_row`，则标记 `use_previous = True`，表示后续将使用前一帧或初始位置。
                *   否则，进行进一步筛选：
                    1.  **条件1 (圆度和 Y 坐标范围)**:
                        *   筛选出圆度与1的绝对差值小于 `self.circularity_deviation_threshold` 的候选点。
                        *   筛选出 Y 坐标偏差 `yrange` 的绝对值小于30像素的候选点。
                        *   找到同时满足这两个条件的候选点。
                    2.  如果满足条件1的候选点数量仍少于 `num_targets_per_row`，则标记 `use_previous = True`。
                    3.  **条件2 (面积优先)**: 对满足条件1的候选点，按其面积降序排列。
                    4.  选取面积最大的前 `num_targets_per_row` 个候选点。
                    5.  **X 坐标排序**: 将这 `num_targets_per_row` 个选定的点按其 X 坐标升序排列，以确保目标点在行内的顺序正确。
                    6.  将最终确定的这些点的 (x, y) 坐标存入当前帧对应的 `self.loc_pt` 位置。标记 `use_previous = False`。
                    7.  **大幅度移动检查 (最后确认)**:
                        *   如果不是第一帧 (`img_idx > 0`)，比较当前帧检测到的 X 坐标与前一帧对应点的 X 坐标。如果任何点的 X 坐标变化超过 `self.target_large_movement`，则认为当前检测可能错误，重新标记 `use_previous = True`。
                        *   如果是第一帧，则比较当前检测到的 X 坐标与初始配置的 X 坐标 (`self.ini_loc_w`)。如果变化过大，则直接使用初始位置，并标记 `use_previous = False`（因为已经修正为初始值）。
            *   **应用 `use_previous` 逻辑**:
                *   如果 `use_previous` 为 `True`：
                    *   若不是第一帧，则将前一帧该行目标点的坐标复制到当前帧。
                    *   若是第一帧，则使用配置中的初始坐标 (`self.ini_loc_w` 和 `self.ini_loc_h[row_idx]`)。

    *   **绘图与视频帧生成**:
        *   `fig.clf()`: 清除 Matplotlib 图形对象，为绘制新一帧做准备。
        *   获取当前帧所有已确定的目标点坐标 `loc_plot_x`, `loc_plot_y`。
        *   **创建子图 (使用 `plt.subplot2grid` 模拟 MATLAB 的布局)**:
            *   **子图1 (左侧，4行1列)**: 显示当前帧的原始彩色图像 (`img_bgr`)，并在其上用亮绿色圆圈标出检测到的目标点 `loc_plot_x`, `loc_plot_y`。
            *   **子图2 (中间，4行1列)**: 在黑色背景上，根据 `loc_plot_x`, `loc_plot_y` 和 `self.pt_map` 绘制蓝色的骨架连线，并用亮绿色圆圈标出目标点。
            *   **子图3 (右上，2行1列)**: 绘制**最后一行的所有目标点**相对于其在第一帧（或初始位置）的 X 方向位移随时间（帧号/帧率）变化的曲线。
            *   **子图4 (右下，2行1列)**: 绘制**第一列的所有目标点**（即每行的第一个点）相对于其在第一帧（或初始位置）的 X 方向位移随时间变化的曲线。
        *   `plt.tight_layout()`: 调整子图间距。
        *   `_get_frame_from_figure(fig)`: 将 Matplotlib 图形对象 `fig` 的当前内容渲染并转换为 OpenCV BGR 格式的图像帧。此方法内部使用 `fig.canvas.print_to_buffer()` 获取 RGBA 图像数据，然后转换为 BGR。
        *   `self.output_video_writer.write(frame_bgr)`: 将生成的 BGR 图像帧写入视频文件。

*   **结束处理**:
    *   `plt.close(fig)`: 关闭 Matplotlib 图形对象。
    *   `self.output_video_writer.release()`: 释放视频写入器，完成视频文件的保存。
    *   打印 "Complete!!" 消息及输出视频的路径。

### 辅助方法

*   `_get_frame_from_figure(fig)`: 如上所述，负责将 Matplotlib 绘图转换为 OpenCV 图像帧。

## 主程序入口 (`if __name__ == '__main__':`)

*   **配置字典 (`config`)**: 在这里定义了程序运行所需的所有参数。
    *   包含一个用于测试的逻辑：如果 `img` 文件夹不存在，则会创建它并生成一些随机的虚拟图像。此时，`config` 中的 `img_size`, `ini_loc_h`, `ini_loc_w` 等参数会使用为虚拟图像设计的值。否则，使用为实际数据（如 MATLAB 示例中的）设计的默认值。**用户应根据自己的实际图像数据调整这些配置。**
*   **实例化与执行**:
    *   创建 `VisionBasedVibrationMeasurement` 类的实例 `analyzer`，传入 `config`。
    *   调用 `analyzer.process_video()` 方法启动整个处理流程。
*   **异常处理**: 使用 `try-except` 块捕获可能发生的 `FileNotFoundError` 和其他一般性错误，并打印错误信息。

该流程通过逐帧分析、目标检测、属性筛选和位置修正，实现了对视频中多个标记点的稳定追踪，并将追踪结果和相关的位移信息可视化输出到视频文件中。
