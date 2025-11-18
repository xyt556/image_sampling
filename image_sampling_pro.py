import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from PIL import Image
import os
from pathlib import Path
import threading

try:
    import rasterio
    from rasterio.transform import xy

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


class ImageSamplingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图像随机采样工具 - 真实坐标版")
        self.root.geometry("900x800")

        self.image_paths = []
        self.image_vars = []
        self.sample_count = 100

        self.setup_ui()

    def setup_ui(self):
        # 主容器
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=BOTH, expand=YES)

        # 标题
        title_label = ttk.Label(
            main_frame,
            text="图像随机采样工具",
            font=("Arial", 20, "bold"),
            bootstyle="primary"
        )
        title_label.pack(pady=(0, 10))

        subtitle_label = ttk.Label(
            main_frame,
            text="支持多种采样策略和坐标输出格式",
            font=("Arial", 10),
            bootstyle="secondary"
        )
        subtitle_label.pack(pady=(0, 15))

        # 文件选择区域
        file_frame = ttk.Labelframe(
            main_frame,
            text="1. 选择图像文件",
            padding=15,
            bootstyle="info"
        )
        file_frame.pack(fill=BOTH, expand=YES, pady=(0, 15))

        # 按钮行
        btn_frame = ttk.Frame(file_frame)
        btn_frame.pack(fill=X)

        self.select_btn = ttk.Button(
            btn_frame,
            text="选择图像文件",
            command=self.select_images,
            bootstyle="info-outline",
            width=15
        )
        self.select_btn.pack(side=LEFT, padx=(0, 10))

        self.select_all_btn = ttk.Button(
            btn_frame,
            text="全选",
            command=self.select_all,
            bootstyle="success-outline",
            width=10
        )
        self.select_all_btn.pack(side=LEFT, padx=(0, 10))

        self.deselect_all_btn = ttk.Button(
            btn_frame,
            text="全不选",
            command=self.deselect_all,
            bootstyle="danger-outline",
            width=10
        )
        self.deselect_all_btn.pack(side=LEFT, padx=(0, 10))

        self.file_count_label = ttk.Label(
            btn_frame,
            text="未选择文件",
            bootstyle="secondary"
        )
        self.file_count_label.pack(side=LEFT)

        # 文件列表
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=BOTH, expand=YES, pady=(10, 0))

        canvas = ttk.Canvas(list_frame, height=120)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)

        self.file_list_frame = ttk.Frame(canvas)

        self.file_list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.file_list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=LEFT, fill=BOTH, expand=YES)
        scrollbar.pack(side=RIGHT, fill=Y)

        # 采样策略选择区域 - 新增
        strategy_frame = ttk.Labelframe(
            main_frame,
            text="2. 采样策略",
            padding=15,
            bootstyle="success"
        )
        strategy_frame.pack(fill=X, pady=(0, 15))

        # 策略选择
        strategy_select_frame = ttk.Frame(strategy_frame)
        strategy_select_frame.pack(fill=X)

        ttk.Label(
            strategy_select_frame,
            text="策略类型:",
            font=("Arial", 10)
        ).pack(side=LEFT, padx=(0, 10))

        self.strategy_var = ttk.StringVar(value="random")

        ttk.Radiobutton(
            strategy_select_frame,
            text="随机采样",
            variable=self.strategy_var,
            value="random",
            command=self.on_strategy_change,
            bootstyle="success-toolbutton"
        ).pack(side=LEFT, padx=(0, 10))

        ttk.Radiobutton(
            strategy_select_frame,
            text="网格采样",
            variable=self.strategy_var,
            value="grid",
            command=self.on_strategy_change,
            bootstyle="success-toolbutton"
        ).pack(side=LEFT, padx=(0, 10))

        ttk.Radiobutton(
            strategy_select_frame,
            text="分层随机",
            variable=self.strategy_var,
            value="stratified",
            command=self.on_strategy_change,
            bootstyle="success-toolbutton"
        ).pack(side=LEFT, padx=(0, 10))

        ttk.Radiobutton(
            strategy_select_frame,
            text="边缘避让",
            variable=self.strategy_var,
            value="edge_avoid",
            command=self.on_strategy_change,
            bootstyle="success-toolbutton"
        ).pack(side=LEFT)

        # 策略说明
        self.strategy_info_frame = ttk.Frame(strategy_frame)
        self.strategy_info_frame.pack(fill=X, pady=(10, 0))

        self.strategy_info_label = ttk.Label(
            self.strategy_info_frame,
            text="• 完全随机分布的采样点",
            bootstyle="secondary",
            font=("Arial", 9),
            justify=LEFT
        )
        self.strategy_info_label.pack(anchor=W)

        # 边缘避让参数（默认隐藏）
        self.edge_param_frame = ttk.Frame(strategy_frame)

        ttk.Label(
            self.edge_param_frame,
            text="边缘距离:",
            font=("Arial", 9)
        ).pack(side=LEFT, padx=(0, 5))

        self.edge_distance_entry = ttk.Entry(
            self.edge_param_frame,
            width=10,
            bootstyle="success"
        )
        self.edge_distance_entry.insert(0, "10")
        self.edge_distance_entry.pack(side=LEFT, padx=(0, 5))

        ttk.Label(
            self.edge_param_frame,
            text="像素",
            bootstyle="secondary",
            font=("Arial", 9)
        ).pack(side=LEFT)

        # 分层采样参数（默认隐藏）
        self.stratified_param_frame = ttk.Frame(strategy_frame)

        ttk.Label(
            self.stratified_param_frame,
            text="网格大小:",
            font=("Arial", 9)
        ).pack(side=LEFT, padx=(0, 5))

        self.grid_rows_entry = ttk.Entry(
            self.stratified_param_frame,
            width=8,
            bootstyle="success"
        )
        self.grid_rows_entry.insert(0, "10")
        self.grid_rows_entry.pack(side=LEFT, padx=(0, 5))

        ttk.Label(
            self.stratified_param_frame,
            text="×",
            font=("Arial", 9)
        ).pack(side=LEFT, padx=(0, 5))

        self.grid_cols_entry = ttk.Entry(
            self.stratified_param_frame,
            width=8,
            bootstyle="success"
        )
        self.grid_cols_entry.insert(0, "10")
        self.grid_cols_entry.pack(side=LEFT, padx=(0, 5))

        ttk.Label(
            self.stratified_param_frame,
            text="(将图像分为网格，每格采样)",
            bootstyle="secondary",
            font=("Arial", 9)
        ).pack(side=LEFT)

        # 采样点设置区域
        sample_frame = ttk.Labelframe(
            main_frame,
            text="3. 采样参数",
            padding=15,
            bootstyle="info"
        )
        sample_frame.pack(fill=X, pady=(0, 15))

        # 采样点数量
        count_frame = ttk.Frame(sample_frame)
        count_frame.pack(fill=X)

        ttk.Label(
            count_frame,
            text="采样点数量:",
            font=("Arial", 10)
        ).pack(side=LEFT, padx=(0, 10))

        self.sample_entry = ttk.Entry(
            count_frame,
            width=15,
            bootstyle="info"
        )
        self.sample_entry.insert(0, "100")
        self.sample_entry.pack(side=LEFT, padx=(0, 10))

        self.sample_hint_label = ttk.Label(
            count_frame,
            text="(建议: 100-10000)",
            bootstyle="secondary",
            font=("Arial", 9)
        )
        self.sample_hint_label.pack(side=LEFT)

        # 波段选择
        band_frame = ttk.Frame(sample_frame)
        band_frame.pack(fill=X, pady=(10, 0))

        ttk.Label(
            band_frame,
            text="读取模式:",
            font=("Arial", 10)
        ).pack(side=LEFT, padx=(0, 10))

        self.band_var = ttk.StringVar(value="grayscale")
        ttk.Radiobutton(
            band_frame,
            text="灰度",
            variable=self.band_var,
            value="grayscale",
            bootstyle="info-toolbutton"
        ).pack(side=LEFT, padx=(0, 10))

        ttk.Radiobutton(
            band_frame,
            text="RGB",
            variable=self.band_var,
            value="rgb",
            bootstyle="info-toolbutton"
        ).pack(side=LEFT)

        # 坐标类型选择
        coord_frame = ttk.Labelframe(
            main_frame,
            text="4. 坐标输出设置",
            padding=15,
            bootstyle="primary"
        )
        coord_frame.pack(fill=X, pady=(0, 15))

        ttk.Label(
            coord_frame,
            text="坐标类型:",
            font=("Arial", 10)
        ).pack(side=LEFT, padx=(0, 10))

        self.coord_var = ttk.StringVar(value="pixel_center")

        ttk.Radiobutton(
            coord_frame,
            text="像素坐标（行列号）",
            variable=self.coord_var,
            value="pixel_index",
            bootstyle="primary-toolbutton"
        ).pack(side=LEFT, padx=(0, 10))

        ttk.Radiobutton(
            coord_frame,
            text="像元中心坐标（X,Y）",
            variable=self.coord_var,
            value="pixel_center",
            bootstyle="primary-toolbutton"
        ).pack(side=LEFT, padx=(0, 10))

        if RASTERIO_AVAILABLE:
            ttk.Radiobutton(
                coord_frame,
                text="地理坐标（需GeoTIFF）",
                variable=self.coord_var,
                value="geo",
                bootstyle="primary-toolbutton"
            ).pack(side=LEFT)
        else:
            ttk.Label(
                coord_frame,
                text="(安装rasterio以支持地理坐标)",
                bootstyle="warning",
                font=("Arial", 9)
            ).pack(side=LEFT)

        # 坐标说明
        coord_info = ttk.Frame(coord_frame)
        coord_info.pack(fill=X, pady=(10, 0))

        info_text = ttk.Label(
            coord_info,
            text="• 像素坐标: 整数行列号 (row, col) • 像元中心: 像素中心点坐标 (x, y) • 地理坐标: 真实地理坐标",
            bootstyle="secondary",
            font=("Arial", 9),
            justify=LEFT
        )
        info_text.pack(anchor=W)

        # 输出设置区域
        output_frame = ttk.Labelframe(
            main_frame,
            text="5. 输出设置",
            padding=15,
            bootstyle="warning"
        )
        output_frame.pack(fill=X, pady=(0, 15))

        output_btn_frame = ttk.Frame(output_frame)
        output_btn_frame.pack(fill=X)

        self.output_btn = ttk.Button(
            output_btn_frame,
            text="选择输出路径",
            command=self.select_output,
            bootstyle="warning-outline",
            width=15
        )
        self.output_btn.pack(side=LEFT, padx=(0, 10))

        self.output_label = ttk.Label(
            output_btn_frame,
            text="默认保存到桌面",
            bootstyle="secondary"
        )
        self.output_label.pack(side=LEFT, fill=X, expand=YES)

        # 开始采样按钮
        self.execute_btn = ttk.Button(
            output_btn_frame,
            text="开始采样",
            command=self.start_sampling,
            bootstyle="success",
            width=15
        )
        self.execute_btn.pack(side=RIGHT)

        self.output_path = None

        # 进度条
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate',
            bootstyle="success-striped"
        )
        self.progress.pack(fill=X, pady=(0, 10))

        # 状态标签
        self.status_label = ttk.Label(
            main_frame,
            text="就绪",
            font=("Arial", 10),
            bootstyle="secondary"
        )
        self.status_label.pack()

    def on_strategy_change(self):
        """策略改变时更新说明和参数"""
        strategy = self.strategy_var.get()

        # 隐藏所有参数框
        self.edge_param_frame.pack_forget()
        self.stratified_param_frame.pack_forget()

        # 根据策略显示说明和参数
        if strategy == "random":
            info = "• 完全随机分布的采样点，适合一般用途"
            self.sample_hint_label.config(text="(建议: 100-10000)")

        elif strategy == "grid":
            info = "• 按固定间隔采样，分布均匀，适合系统性分析"
            self.sample_hint_label.config(text="(采样点数 ≈ √n × √n)")

        elif strategy == "stratified":
            info = "• 将图像分区后在每个区域随机采样，兼顾随机性和分布均匀性"
            self.sample_hint_label.config(text="(每格采样点数 ≈ 总数/格数)")
            self.stratified_param_frame.pack(fill=X, pady=(10, 0))

        elif strategy == "edge_avoid":
            info = "• 避免在图像边缘采样，减少边界效应影响"
            self.sample_hint_label.config(text="(建议: 100-10000)")
            self.edge_param_frame.pack(fill=X, pady=(10, 0))

        self.strategy_info_label.config(text=info)

    def generate_sample_points(self, height, width, count):
        """根据选择的策略生成采样点"""
        strategy = self.strategy_var.get()

        if strategy == "random":
            # 随机采样
            rows = np.random.randint(0, height, count)
            cols = np.random.randint(0, width, count)

        elif strategy == "grid":
            # 网格采样
            grid_size = int(np.sqrt(count))
            row_step = height / (grid_size + 1)
            col_step = width / (grid_size + 1)

            rows = []
            cols = []
            for i in range(1, grid_size + 1):
                for j in range(1, grid_size + 1):
                    rows.append(int(i * row_step))
                    cols.append(int(j * col_step))

            rows = np.array(rows[:count])
            cols = np.array(cols[:count])

        elif strategy == "stratified":
            # 分层随机采样
            try:
                grid_rows = int(self.grid_rows_entry.get())
                grid_cols = int(self.grid_cols_entry.get())
            except ValueError:
                grid_rows = grid_cols = 10

            cell_height = height / grid_rows
            cell_width = width / grid_cols
            samples_per_cell = max(1, count // (grid_rows * grid_cols))

            rows = []
            cols = []

            for i in range(grid_rows):
                for j in range(grid_cols):
                    for _ in range(samples_per_cell):
                        r = int(i * cell_height + np.random.random() * cell_height)
                        c = int(j * cell_width + np.random.random() * cell_width)
                        rows.append(min(r, height - 1))
                        cols.append(min(c, width - 1))

            rows = np.array(rows[:count])
            cols = np.array(cols[:count])

        elif strategy == "edge_avoid":
            # 边缘避让采样
            try:
                edge_dist = int(self.edge_distance_entry.get())
            except ValueError:
                edge_dist = 10

            safe_height = max(1, height - 2 * edge_dist)
            safe_width = max(1, width - 2 * edge_dist)

            rows = np.random.randint(edge_dist, edge_dist + safe_height, count)
            cols = np.random.randint(edge_dist, edge_dist + safe_width, count)

        else:
            # 默认随机
            rows = np.random.randint(0, height, count)
            cols = np.random.randint(0, width, count)

        return rows, cols

    def select_images(self):
        """选择图像文件"""
        files = filedialog.askopenfilenames(
            title="选择图像文件",
            filetypes=[
                ("所有支持格式", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("GeoTIFF文件", "*.tif *.tiff"),
                ("普通图像", "*.png *.jpg *.jpeg *.bmp"),
                ("所有文件", "*.*")
            ]
        )

        if files:
            self.image_paths = list(files)
            self.image_vars = []

            for widget in self.file_list_frame.winfo_children():
                widget.destroy()

            for idx, file in enumerate(files):
                var = ttk.BooleanVar(value=True)
                self.image_vars.append(var)

                frame = ttk.Frame(self.file_list_frame)
                frame.pack(fill=X, pady=2)

                cb = ttk.Checkbutton(
                    frame,
                    text=f"{idx + 1}. {os.path.basename(file)}",
                    variable=var,
                    bootstyle="info-round-toggle"
                )
                cb.pack(side=LEFT, anchor=W)

                # 检测地理信息
                try:
                    if RASTERIO_AVAILABLE and file.lower().endswith(('.tif', '.tiff')):
                        try:
                            with rasterio.open(file) as src:
                                if src.crs:
                                    geo_label = ttk.Label(
                                        frame,
                                        text=f"[{src.crs}]",
                                        bootstyle="success",
                                        font=("Arial", 8)
                                    )
                                    geo_label.pack(side=RIGHT, padx=(5, 0))
                        except Exception:
                            pass

                    with Image.open(file) as img:
                        size_label = ttk.Label(
                            frame,
                            text=f"({img.width}×{img.height})",
                            bootstyle="secondary",
                            font=("Arial", 9)
                        )
                        size_label.pack(side=RIGHT)
                except Exception as e:
                    error_label = ttk.Label(
                        frame,
                        text="(无法读取)",
                        bootstyle="danger",
                        font=("Arial", 9)
                    )
                    error_label.pack(side=RIGHT)

            selected_count = sum(var.get() for var in self.image_vars)
            self.file_count_label.config(text=f"已选择 {selected_count}/{len(files)} 个文件")

    def select_all(self):
        """全选"""
        for var in self.image_vars:
            var.set(True)
        if self.image_vars:
            self.file_count_label.config(
                text=f"已选择 {len(self.image_vars)}/{len(self.image_vars)} 个文件"
            )

    def deselect_all(self):
        """全不选"""
        for var in self.image_vars:
            var.set(False)
        if self.image_vars:
            self.file_count_label.config(
                text=f"已选择 0/{len(self.image_vars)} 个文件"
            )

    def select_output(self):
        """选择输出路径"""
        file_path = filedialog.asksaveasfilename(
            title="保存CSV文件",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")],
            initialfile="sampling_results.csv"
        )

        if file_path:
            self.output_path = file_path
            self.output_label.config(text=f"保存到: {os.path.basename(file_path)}")

    def start_sampling(self):
        """开始采样"""
        if not self.image_paths:
            messagebox.showwarning("警告", "请先选择图像文件！")
            return

        selected_count = sum(var.get() for var in self.image_vars)
        if selected_count == 0:
            messagebox.showwarning("警告", "请至少选择一个图像文件！")
            return

        try:
            self.sample_count = int(self.sample_entry.get())
            if self.sample_count <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "请输入有效的采样点数量！")
            return

        # 检查地理坐标模式
        if self.coord_var.get() == "geo" and not RASTERIO_AVAILABLE:
            messagebox.showerror("错误", "需要安装 rasterio 才能使用地理坐标模式！\n\n安装命令: pip install rasterio")
            return

        # 禁用按钮
        self.execute_btn.config(state="disabled")
        self.select_btn.config(state="disabled")
        self.output_btn.config(state="disabled")
        self.select_all_btn.config(state="disabled")
        self.deselect_all_btn.config(state="disabled")

        self.progress.start()

        thread = threading.Thread(target=self.perform_sampling)
        thread.start()

    def get_coordinates(self, rows, cols, img_path=None, transform=None):
        """根据选择的坐标类型计算坐标"""
        coord_type = self.coord_var.get()

        if coord_type == "pixel_index":
            return rows, cols, "row", "col"

        elif coord_type == "pixel_center":
            x = cols + 0.5
            y = rows + 0.5
            return x, y, "x_pixel", "y_pixel"

        elif coord_type == "geo":
            if transform is None:
                x = cols + 0.5
                y = rows + 0.5
                return x, y, "x_pixel", "y_pixel"
            else:
                xs, ys = [], []
                for row, col in zip(rows, cols):
                    x, y = xy(transform, row + 0.5, col + 0.5)
                    xs.append(x)
                    ys.append(y)
                return np.array(xs), np.array(ys), "x_geo", "y_geo"

        return rows, cols, "row", "col"

    def perform_sampling(self):
        """执行采样操作"""
        try:
            self.update_status("正在读取图像...")

            selected_images = []
            image_names = []
            image_sizes = []
            image_transforms = []

            for img_path, var in zip(self.image_paths, self.image_vars):
                if not var.get():
                    continue

                try:
                    transform = None

                    if RASTERIO_AVAILABLE and img_path.lower().endswith(('.tif', '.tiff')):
                        try:
                            with rasterio.open(img_path) as src:
                                if src.crs:
                                    transform = src.transform
                                    self.update_status(f"检测到地理信息: {Path(img_path).name}")
                        except Exception:
                            pass

                    img = Image.open(img_path)

                    if self.band_var.get() == "grayscale":
                        img = img.convert('L')
                    else:
                        img = img.convert('RGB')

                    img_array = np.array(img)
                    selected_images.append(img_array)
                    image_names.append(Path(img_path).stem)
                    image_sizes.append(img_array.shape[:2])
                    image_transforms.append(transform)

                    self.update_status(f"已读取: {Path(img_path).name}")
                except Exception as e:
                    self.update_status(f"读取失败: {Path(img_path).name} - {str(e)}")
                    continue

            if not selected_images:
                raise Exception("没有成功读取任何图像！")

            min_height = min(size[0] for size in image_sizes)
            min_width = min(size[1] for size in image_sizes)

            self.update_status(f"图像有效区域: {min_width}×{min_height}")

            strategy_name = {
                "random": "随机采样",
                "grid": "网格采样",
                "stratified": "分层随机采样",
                "edge_avoid": "边缘避让采样"
            }
            self.update_status(f"正在使用{strategy_name[self.strategy_var.get()]}生成 {self.sample_count} 个采样点...")

            # 生成采样点
            np.random.seed(42)
            sample_rows, sample_cols = self.generate_sample_points(min_height, min_width, self.sample_count)

            actual_count = len(sample_rows)
            self.update_status(f"实际生成 {actual_count} 个采样点")

            self.update_status("正在计算坐标...")

            x_coords, y_coords, x_label, y_label = self.get_coordinates(
                sample_rows, sample_cols,
                self.image_paths[0] if self.image_paths else None,
                image_transforms[0] if image_transforms else None
            )

            data = {
                x_label: x_coords,
                y_label: y_coords
            }

            if self.coord_var.get() == "geo":
                data['row'] = sample_rows
                data['col'] = sample_cols

            self.update_status("正在提取像元值...")

            for img_name, img_array in zip(image_names, selected_images):
                if len(img_array.shape) == 2:
                    values = img_array[sample_rows, sample_cols]
                    data[img_name] = values
                else:
                    for band_idx, band_name in enumerate(['R', 'G', 'B']):
                        values = img_array[sample_rows, sample_cols, band_idx]
                        data[f"{img_name}_{band_name}"] = values

            df = pd.DataFrame(data)

            if self.output_path:
                output_file = self.output_path
            else:
                desktop = Path.home() / "Desktop"
                output_file = desktop / "sampling_results.csv"

            self.update_status("正在保存结果...")

            df.to_csv(output_file, index=False, encoding='utf-8-sig')

            coord_type_msg = {
                "pixel_index": "像素坐标（行列号）",
                "pixel_center": "像元中心坐标",
                "geo": "地理坐标"
            }

            self.update_status(f"完成！已保存到: {output_file}")

            self.root.after(0, lambda: messagebox.showinfo(
                "成功",
                f"采样完成！\n\n"
                f"采样策略: {strategy_name[self.strategy_var.get()]}\n"
                f"采样点数: {actual_count}\n"
                f"处理图像: {len(selected_images)}\n"
                f"坐标类型: {coord_type_msg[self.coord_var.get()]}\n"
                f"有效区域: {min_width}×{min_height}\n"
                f"输出文件: {output_file}"
            ))

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            self.update_status(f"错误: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"采样过程出错:\n{str(e)}"))

        finally:
            self.root.after(0, self.reset_ui)

    def update_status(self, message):
        """更新状态标签"""
        self.root.after(0, lambda: self.status_label.config(text=message))

    def reset_ui(self):
        """重置UI状态"""
        self.progress.stop()
        self.execute_btn.config(state="normal")
        self.select_btn.config(state="normal")
        self.output_btn.config(state="normal")
        self.select_all_btn.config(state="normal")
        self.deselect_all_btn.config(state="normal")


def main():
    root = ttk.Window(themename="cosmo")
    app = ImageSamplingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()