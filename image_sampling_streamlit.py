import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
from pathlib import Path
import tempfile
import os

try:
    import rasterio
    from rasterio.transform import xy

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# 页面配置
st.set_page_config(
    page_title="图像随机采样工具",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """初始化session state"""
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'file_selection' not in st.session_state:
        st.session_state.file_selection = {}
    if 'sampling_done' not in st.session_state:
        st.session_state.sampling_done = False
    if 'result_df' not in st.session_state:
        st.session_state.result_df = None
    if 'result_info' not in st.session_state:
        st.session_state.result_info = {}


def generate_sample_points(height, width, count, strategy, **params):
    """根据选择的策略生成采样点"""
    if strategy == "random":
        rows = np.random.randint(0, height, count)
        cols = np.random.randint(0, width, count)

    elif strategy == "grid":
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
        grid_rows = params.get('grid_rows', 10)
        grid_cols = params.get('grid_cols', 10)

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
        edge_dist = params.get('edge_distance', 10)
        safe_height = max(1, height - 2 * edge_dist)
        safe_width = max(1, width - 2 * edge_dist)

        rows = np.random.randint(edge_dist, edge_dist + safe_height, count)
        cols = np.random.randint(edge_dist, edge_dist + safe_width, count)

    else:
        rows = np.random.randint(0, height, count)
        cols = np.random.randint(0, width, count)

    return rows, cols


def get_coordinates(rows, cols, coord_type, transform=None):
    """根据选择的坐标类型计算坐标"""
    if coord_type == "像素坐标（行列号）":
        return rows, cols, "row", "col"

    elif coord_type == "像元中心坐标（X,Y）":
        x = cols + 0.5
        y = rows + 0.5
        return x, y, "x_pixel", "y_pixel"

    elif coord_type == "地理坐标（需GeoTIFF）":
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


def perform_sampling(uploaded_files, selected_files, sample_count, strategy,
                     coord_type, band_mode, strategy_params):
    """执行采样操作"""
    try:
        # 读取选中的图像
        selected_images = []
        image_names = []
        image_sizes = []
        image_transforms = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_files = len(selected_files)

        for idx, (file_obj, file_name) in enumerate(uploaded_files):
            if file_name not in selected_files:
                continue

            status_text.text(f"正在读取: {file_name} ({idx + 1}/{total_files})")
            progress_bar.progress((idx + 1) / total_files * 0.3)

            transform = None

            # 检查是否为GeoTIFF
            if RASTERIO_AVAILABLE and file_name.lower().endswith(('.tif', '.tiff')):
                try:
                    # 保存到临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as tmp_file:
                        tmp_file.write(file_obj.getvalue())
                        tmp_path = tmp_file.name

                    with rasterio.open(tmp_path) as src:
                        if src.crs:
                            transform = src.transform

                    os.unlink(tmp_path)
                except Exception:
                    pass

            # 读取图像
            img = Image.open(file_obj)

            if band_mode == "grayscale":
                img = img.convert('L')
            else:
                img = img.convert('RGB')

            img_array = np.array(img)
            selected_images.append(img_array)
            image_names.append(Path(file_name).stem)
            image_sizes.append(img_array.shape[:2])
            image_transforms.append(transform)

        if not selected_images:
            st.error("没有成功读取任何图像！")
            return None, None

        # 找到最小尺寸
        min_height = min(size[0] for size in image_sizes)
        min_width = min(size[1] for size in image_sizes)

        status_text.text(f"图像有效区域: {min_width}×{min_height}")
        progress_bar.progress(0.4)

        # 生成采样点
        status_text.text(f"正在生成采样点...")
        np.random.seed(42)
        sample_rows, sample_cols = generate_sample_points(
            min_height, min_width, sample_count, strategy, **strategy_params
        )

        actual_count = len(sample_rows)
        progress_bar.progress(0.5)

        # 计算坐标
        status_text.text("正在计算坐标...")
        x_coords, y_coords, x_label, y_label = get_coordinates(
            sample_rows, sample_cols, coord_type,
            image_transforms[0] if image_transforms else None
        )

        data = {
            x_label: x_coords,
            y_label: y_coords
        }

        if coord_type == "地理坐标（需GeoTIFF）" and image_transforms[0] is not None:
            data['row'] = sample_rows
            data['col'] = sample_cols

        progress_bar.progress(0.6)

        # 提取像元值
        status_text.text("正在提取像元值...")
        for img_name, img_array in zip(image_names, selected_images):
            if len(img_array.shape) == 2:  # 灰度图
                values = img_array[sample_rows, sample_cols]
                data[img_name] = values
            else:  # RGB图
                for band_idx, band_name in enumerate(['R', 'G', 'B']):
                    values = img_array[sample_rows, sample_cols, band_idx]
                    data[f"{img_name}_{band_name}"] = values

        progress_bar.progress(0.9)

        # 创建DataFrame
        df = pd.DataFrame(data)

        # 结果信息
        result_info = {
            'strategy': strategy,
            'sample_count': actual_count,
            'image_count': len(selected_images),
            'coord_type': coord_type,
            'valid_area': f"{min_width}×{min_height}",
            'image_names': image_names
        }

        progress_bar.progress(1.0)
        status_text.text("采样完成！")

        return df, result_info

    except Exception as e:
        st.error(f"采样过程出错: {str(e)}")
        return None, None


def main():
    initialize_session_state()

    # 标题
    st.markdown('<div class="main-header">🖼️ 图像随机采样工具</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">支持多种采样策略和坐标输出格式</div>', unsafe_allow_html=True)

    # 侧边栏配置
    with st.sidebar:
        st.header("⚙️ 采样配置")

        # 采样策略
        st.subheader("1️⃣ 采样策略")
        strategy = st.radio(
            "选择策略",
            ["随机采样", "网格采样", "分层随机", "边缘避让"],
            help="选择不同的采样策略"
        )

        # 策略说明
        strategy_info = {
            "随机采样": "完全随机分布的采样点，适合一般用途",
            "网格采样": "按固定间隔采样，分布均匀，适合系统性分析",
            "分层随机": "将图像分区后在每个区域随机采样，兼顾随机性和分布均匀性",
            "边缘避让": "避免在图像边缘采样，减少边界效应影响"
        }
        st.info(strategy_info[strategy])

        # 策略参数
        strategy_params = {}
        if strategy == "分层随机":
            col1, col2 = st.columns(2)
            with col1:
                strategy_params['grid_rows'] = st.number_input(
                    "网格行数", min_value=2, max_value=50, value=10
                )
            with col2:
                strategy_params['grid_cols'] = st.number_input(
                    "网格列数", min_value=2, max_value=50, value=10
                )
        elif strategy == "边缘避让":
            strategy_params['edge_distance'] = st.number_input(
                "边缘距离（像素）", min_value=1, max_value=100, value=10
            )

        st.divider()

        # 采样参数
        st.subheader("2️⃣ 采样参数")
        sample_count = st.number_input(
            "采样点数量",
            min_value=1,
            max_value=100000,
            value=100,
            step=10,
            help="建议范围: 100-10000"
        )

        band_mode = st.radio(
            "读取模式",
            ["grayscale", "rgb"],
            format_func=lambda x: "灰度" if x == "grayscale" else "RGB"
        )

        st.divider()

        # 坐标输出设置
        st.subheader("3️⃣ 坐标输出")
        coord_options = ["像素坐标（行列号）", "像元中心坐标（X,Y）"]
        if RASTERIO_AVAILABLE:
            coord_options.append("地理坐标（需GeoTIFF）")

        coord_type = st.radio(
            "坐标类型",
            coord_options,
            index=1
        )

        if not RASTERIO_AVAILABLE:
            st.warning("安装 rasterio 以支持地理坐标\n```pip install rasterio```")

    # 主界面
    # 文件上传
    st.markdown('<div class="section-header">📁 1. 选择图像文件</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "上传图像文件",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
        accept_multiple_files=True,
        help="支持 PNG, JPG, BMP, TIFF 格式"
    )

    if uploaded_files:
        st.success(f"已上传 {len(uploaded_files)} 个文件")

        # 文件列表和选择
        col1, col2, col3 = st.columns([1, 6, 1])
        with col1:
            if st.button("🔘 全选", use_container_width=True):
                for file in uploaded_files:
                    st.session_state.file_selection[file.name] = True
        with col3:
            if st.button("⭕ 全不选", use_container_width=True):
                for file in uploaded_files:
                    st.session_state.file_selection[file.name] = False

        # 显示文件列表
        with st.expander("📋 查看文件列表", expanded=True):
            for idx, file in enumerate(uploaded_files):
                col1, col2, col3 = st.columns([0.5, 6, 2])

                with col1:
                    if file.name not in st.session_state.file_selection:
                        st.session_state.file_selection[file.name] = True

                    st.session_state.file_selection[file.name] = st.checkbox(
                        "",
                        value=st.session_state.file_selection[file.name],
                        key=f"checkbox_{idx}_{file.name}"
                    )

                with col2:
                    st.text(f"{idx + 1}. {file.name}")

                with col3:
                    try:
                        img = Image.open(file)
                        st.text(f"{img.width}×{img.height}")
                        file.seek(0)  # 重置文件指针
                    except:
                        st.text("无法读取")

        # 显示选中文件数量
        selected_count = sum(st.session_state.file_selection.values())
        st.info(f"✅ 已选择 {selected_count} / {len(uploaded_files)} 个文件")

        st.divider()

        # 采样按钮
        st.markdown('<div class="section-header">🎯 2. 开始采样</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            start_sampling = st.button(
                "🚀 开始采样",
                type="primary",
                use_container_width=True
            )

        if start_sampling:
            if selected_count == 0:
                st.error("❌ 请至少选择一个图像文件！")
            else:
                # 准备文件数据
                file_data = [(file, file.name) for file in uploaded_files]
                selected_files = [name for name, selected in st.session_state.file_selection.items() if selected]

                # 执行采样
                with st.spinner("正在采样，请稍候..."):
                    df, info = perform_sampling(
                        file_data,
                        selected_files,
                        sample_count,
                        strategy,
                        coord_type,
                        band_mode,
                        strategy_params
                    )

                if df is not None:
                    st.session_state.result_df = df
                    st.session_state.result_info = info
                    st.session_state.sampling_done = True
                    st.rerun()

        # 显示结果
        if st.session_state.sampling_done and st.session_state.result_df is not None:
            st.divider()
            st.markdown('<div class="section-header">📊 3. 采样结果</div>', unsafe_allow_html=True)

            info = st.session_state.result_info

            # 结果摘要
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("采样策略", info['strategy'])
            with col2:
                st.metric("采样点数", info['sample_count'])
            with col3:
                st.metric("处理图像", info['image_count'])
            with col4:
                st.metric("有效区域", info['valid_area'])

            # 显示数据预览
            st.subheader("📝 数据预览")
            st.dataframe(st.session_state.result_df.head(20), use_container_width=True)

            # 统计信息
            with st.expander("📈 数据统计"):
                st.write(st.session_state.result_df.describe())

            # 下载按钮
            st.subheader("💾 下载结果")

            csv = st.session_state.result_df.to_csv(index=False, encoding='utf-8-sig')

            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                st.download_button(
                    label="📥 下载 CSV 文件",
                    data=csv,
                    file_name="sampling_results.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )

            # 重新采样按钮
            st.divider()
            col1, col2, col3 = st.columns([2, 1, 2])
            with col2:
                if st.button("🔄 重新采样", use_container_width=True):
                    st.session_state.sampling_done = False
                    st.session_state.result_df = None
                    st.session_state.result_info = {}
                    st.rerun()

    else:
        st.info("👆 请在上方上传图像文件以开始")

        # 使用说明
        with st.expander("📖 使用说明"):
            st.markdown("""
            ### 使用步骤

            1. **上传图像文件**
               - 支持 PNG, JPG, BMP, TIFF 等格式
               - 可同时上传多个文件

            2. **选择图像**
               - 勾选需要处理的图像
               - 使用"全选"/"全不选"快速操作

            3. **配置参数**（在左侧边栏）
               - 选择采样策略
               - 设置采样点数量
               - 选择读取模式（灰度/RGB）
               - 选择坐标输出类型

            4. **开始采样**
               - 点击"开始采样"按钮
               - 等待处理完成

            5. **下载结果**
               - 预览采样数据
               - 下载 CSV 文件

            ### 采样策略说明

            - **随机采样**: 完全随机分布，适合一般用途
            - **网格采样**: 均匀分布，适合系统性分析
            - **分层随机**: 分区采样，兼顾随机性和均匀性
            - **边缘避让**: 避免边缘区域，减少边界效应

            ### 坐标类型说明

            - **像素坐标**: 整数行列号 (row, col)
            - **像元中心**: 像素中心点坐标 (x, y)
            - **地理坐标**: 真实地理坐标（需要 GeoTIFF 格式）
            """)


if __name__ == "__main__":
    main()