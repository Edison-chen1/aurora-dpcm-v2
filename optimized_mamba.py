import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import time
import cupy as cp
import math
from torch.utils.cpp_extension import load
import astropy.io.fits as fits
from typing import Optional, Tuple
import os
from dataclasses import dataclass

# 定义设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# 配置设置
@dataclass
class MambaDPCMConfig:
    d_model: int = 64  # 模型维度
    n_layer: int = 4  # Mamba块数量
    d_state: int = 16  # SSM状态大小
    d_conv: int = 4  # 卷积核大小
    expand: int = 2  # 扩展因子
    dt_min: float = 0.001  # 最小delta值
    dt_max: float = 0.1  # 最大delta值
    dt_init: str = "random"  # 初始化方式
    dt_scale: float = 1.0  # 缩放因子
    dt_init_floor: float = 1e-4  # 最小初始值
    bias: bool = True  # 是否使用偏置
    conv_bias: bool = True  # 是否使用卷积偏置

    # DPCM特定参数
    block_size: int = 256  # 处理块大小
    pred_order: int = 11  # 预测阶数 (N)
    eq_count: int = 7  # 方程数量 (M)
    threshold_init: int = 13  # 残差阈值初始值

    # GPU优化参数
    streams_per_device: int = 4  # 每个GPU的CUDA流数量
    devices: int = 1  # 使用的GPU数量
    shared_mem_size: int = 48 * 1024  # 共享内存大小 (默认48KB)


# 选择性状态空间模型(SSM)核心实现
class SSM(nn.Module):
    def __init__(self, config, d_model):
        super().__init__()
        self.d_model = d_model
        self.d_state = config.d_state

        # 离散化参数
        self.dt_min = config.dt_min
        self.dt_max = config.dt_max
        self.dt_init = config.dt_init
        self.dt_scale = config.dt_scale
        self.dt_init_floor = config.dt_init_floor

        # SSM参数
        # 初始化A, B, C, D矩阵
        self.A = nn.Parameter(torch.randn(self.d_model, self.d_state, self.d_state))
        self.B = nn.Parameter(torch.randn(self.d_model, self.d_state, 1))
        self.C = nn.Parameter(torch.randn(self.d_model, 1, self.d_state))
        self.D = nn.Parameter(torch.zeros(self.d_model))

        # 数据依赖的参数Δ (delta)
        self.delta = nn.Parameter(torch.Tensor(self.d_model))
        self._init_delta()

        # 参数投影层
        self.A_proj = nn.Linear(d_model, self.d_state * self.d_state)
        self.B_proj = nn.Linear(d_model, self.d_state)
        self.C_proj = nn.Linear(d_model, self.d_state)
        self.D_proj = nn.Linear(d_model, 1)

    def _init_delta(self):
        if self.dt_init == "random":
            # 在dt_min和dt_max之间随机初始化delta
            nn.init.uniform_(self.delta, a=math.log(self.dt_min), b=math.log(self.dt_max))
            self.delta.data = torch.exp(self.delta.data) * self.dt_scale
        else:
            # 常数初始化
            val = float(self.dt_init) * self.dt_scale
            with torch.no_grad():
                self.delta.fill_(val)
        self.delta.data = torch.clamp(self.delta.data, min=self.dt_init_floor)

    def forward(self, u, state=None):
        """
        SSM前向传播
        u: 输入张量, 形状为 [batch, seq_len, d_model]
        state: 可选的初始状态
        """
        batch, seq_len, d_model = u.shape

        # 状态初始化
        if state is None:
            state = torch.zeros(batch, self.d_state, device=u.device)

        # 数据依赖参数 (选择性机制)
        delta = torch.sigmoid(u @ self.delta.view(-1, 1))  # [batch, seq_len, 1]

        # 获取离散化参数
        A_discrete = torch.matrix_exp(self.A * delta.unsqueeze(-1))

        outputs = []
        for t in range(seq_len):
            # 输入投影 - 选择性机制
            B_t = self.B_proj(u[:, t])  # [batch, d_state]
            C_t = self.C_proj(u[:, t])  # [batch, d_state]
            D_t = self.D_proj(u[:, t])  # [batch, 1]

            # 状态更新
            state = A_discrete[:, t] @ state.unsqueeze(-1) + B_t.unsqueeze(-1)  # 矩阵乘法
            state = state.squeeze(-1)

            # 输出计算
            y = (C_t * state).sum(dim=1, keepdim=True) + D_t
            outputs.append(y)

        return torch.cat(outputs, dim=1).view(batch, seq_len, -1)


# Mamba块实现
class MambaBlock(nn.Module):
    def __init__(self, config, d_model=None):
        super().__init__()
        self.config = config
        self.d_model = d_model or config.d_model
        self.expand = config.expand

        # 归一化层
        self.norm = nn.LayerNorm(self.d_model)

        # 局部上下文卷积
        self.conv = nn.Conv1d(
            in_channels=self.d_model * self.expand,
            out_channels=self.d_model * self.expand,
            kernel_size=config.d_conv,
            padding=config.d_conv - 1,
            groups=self.d_model * self.expand,
            bias=config.conv_bias
        )

        # 线性上投影
        self.in_proj = nn.Linear(
            self.d_model,
            self.d_model * self.expand * 2,  # 2用于门控
            bias=config.bias
        )

        # SSM层
        self.ssm = SSM(config, d_model=self.d_model * self.expand)

        # 输出投影
        self.out_proj = nn.Linear(
            self.d_model * self.expand,
            self.d_model,
            bias=config.bias
        )

    def forward(self, x):
        # 输入: [batch, seq_len, d_model]
        residual = x
        x = self.norm(x)

        # 输入投影与门控
        x_proj = self.in_proj(x)  # [batch, seq_len, 2*d_model*expand]
        x_proj_1, x_proj_2 = x_proj.chunk(2, dim=-1)

        # 卷积处理
        x_conv = self.conv(x_proj_1.transpose(1, 2))
        x_conv = x_conv[:, :, :seq_len].transpose(1, 2)

        # SSM处理
        x_ssm = self.ssm(x_conv)

        # SiLU激活和门控
        x_silu = F.silu(x_ssm)
        x_gated = x_silu * x_proj_2

        # 输出投影和残差连接
        return self.out_proj(x_gated) + residual


# Mamba模型主体
class MambaModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 输入嵌入
        self.embed = nn.Linear(1, config.d_model)

        # Mamba块堆叠
        self.layers = nn.ModuleList([
            MambaBlock(config) for _ in range(config.n_layer)
        ])

        # 输出层
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)

    def forward(self, x):
        """
        前向传播
        x: 输入序列 [batch, seq_len]
        """
        # 调整输入维度并嵌入
        x = x.unsqueeze(-1)  # [batch, seq_len, 1]
        x = self.embed(x)

        # 通过所有Mamba块
        for layer in self.layers:
            x = layer(x)

        # 输出层
        x = self.norm(x)
        return self.head(x)


# 动态预测系数生成模块
class DynamicPredictionCoefficients(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pred_order = config.pred_order

        # 使用Mamba模型生成动态预测系数
        self.mamba = MambaModel(config)

        # 映射到预测系数
        self.coef_proj = nn.Linear(config.d_model, self.pred_order)

    def forward(self, history_pixels):
        """
        根据历史像素生成动态预测系数
        history_pixels: 历史像素值 [batch, pred_window]
        """
        # 使用Mamba处理历史像素
        features = self.mamba(history_pixels)

        # 生成预测系数
        coefs = self.coef_proj(features.mean(dim=1))
        return coefs


# 自适应残差阈值生成模块
class AdaptiveThresholdGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 小型Mamba模型处理残差分布
        small_config = MambaDPCMConfig(
            d_model=32,
            n_layer=2,
            d_state=8
        )
        self.mamba = MambaModel(small_config)

        # 映射到双阈值 (T-, T+)
        self.threshold_proj = nn.Linear(small_config.d_model, 2)

        # 初始化阈值
        self.register_buffer('base_threshold',
                             torch.tensor([config.threshold_init], dtype=torch.float))

    def forward(self, residual_histogram):
        """
        根据残差分布直方图生成自适应阈值
        residual_histogram: 残差直方图 [batch, bins]
        """
        # 使用Mamba处理残差直方图
        features = self.mamba(residual_histogram)

        # 生成阈值偏移量
        threshold_offsets = self.threshold_proj(features.mean(dim=1))

        # 基础阈值加上预测的偏移量
        T_minus = self.base_threshold - torch.abs(threshold_offsets[:, 0:1])
        T_plus = self.base_threshold + torch.abs(threshold_offsets[:, 1:2])

        return T_minus, T_plus


# GPU优化的并行扫描算法 (基于CUDA实现)
class ParallelScan:
    """GPU优化的并行扫描算法，用于加速矩阵运算"""

    def __init__(self, config):
        self.config = config
        self.block_size = config.block_size
        self.shared_mem_size = config.shared_mem_size

        # 编译CUDA核心
        self.cuda_module = self._compile_cuda_kernels()

        # 初始化CUDA流
        self.streams = [torch.cuda.Stream() for _ in range(config.streams_per_device)]

    def _compile_cuda_kernels(self):
        """编译CUDA核心函数"""
        cuda_code = """
        #include <cuda_runtime.h>

        // 并行矩阵乘法核心函数
        extern "C" __global__ void matrix_mul_kernel(float* A, float* B, float* C, 
                                                  int M, int N, int K) {
            // 块索引
            int bx = blockIdx.x;
            int by = blockIdx.y;

            // 线程索引
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            // 共享内存声明
            extern __shared__ float shared_mem[];
            float* As = &shared_mem[0];
            float* Bs = &shared_mem[256]; // 假设使用16x16的块

            // 计算全局索引
            int row = by * 16 + ty;
            int col = bx * 16 + tx;

            // 累加器
            float sum = 0.0f;

            // 遍历A和B的分块
            for (int i = 0; i < (K + 15) / 16; ++i) {
                // 加载A分块到共享内存
                if (row < M && i * 16 + tx < K)
                    As[ty * 16 + tx] = A[row * K + i * 16 + tx];
                else
                    As[ty * 16 + tx] = 0.0f;

                // 加载B分块到共享内存
                if (i * 16 + ty < K && col < N)
                    Bs[ty * 16 + tx] = B[(i * 16 + ty) * N + col];
                else
                    Bs[ty * 16 + tx] = 0.0f;

                // 同步以确保数据加载完成
                __syncthreads();

                // 计算当前分块的乘积
                for (int k = 0; k < 16; ++k)
                    sum += As[ty * 16 + k] * Bs[k * 16 + tx];

                // 同步以确保计算完成
                __syncthreads();
            }

            // 写入结果
            if (row < M && col < N)
                C[row * N + col] = sum;
        }

        // CT*C计算的专用核心函数
        extern "C" __global__ void ctc_kernel(float* C, float* result, int M, int N) {
            // 块索引
            int bx = blockIdx.x;
            int by = blockIdx.y;

            // 线程索引
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            // 共享内存
            extern __shared__ float shared_mem[];
            float* Cs = &shared_mem[0];

            // 计算全局索引
            int row = by * 16 + ty;
            int col = bx * 16 + tx;

            // 加载C子矩阵到共享内存
            if (row < M && col < N)
                Cs[ty * 16 + tx] = C[row * N + col];
            else
                Cs[ty * 16 + tx] = 0.0f;

            __syncthreads();

            // 计算C^T * C的一个元素
            if (row < N && col < N) {
                float sum = 0.0f;
                for (int i = 0; i < M; ++i) {
                    if (i < M && row < N)
                        sum += C[i * N + row] * C[i * N + col];
                }
                result[row * N + col] = sum;
            }
        }

        // 残差计算核心函数
        extern "C" __global__ void residual_kernel(float* original, float* predicted, 
                                               float* residual, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                residual[idx] = original[idx] - predicted[idx];
            }
        }

        // 双阈值编码核心函数
        extern "C" __global__ void threshold_encode_kernel(float* residual, int* encoded,
                                                      float T_minus, float T_plus, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                float res = residual[idx];
                if (res >= T_minus && res <= T_plus) {
                    // 在阈值范围内，使用范围编码
                    encoded[idx] = static_cast<int>(res) + 32768; // 偏移以处理负值
                } else {
                    // 超出阈值，标记为特殊值
                    encoded[idx] = (res < T_minus) ? -1 : -2;
                }
            }
        }
        """

        # 使用cupy编译CUDA代码
        module = cp.RawModule(code=cuda_code)
        return module

    def matrix_multiply(self, A, B):
        """
        优化的矩阵乘法实现
        A: 第一个矩阵 [M, K]
        B: 第二个矩阵 [K, N]
        返回: C = A * B [M, N]
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, "矩阵维度不匹配"

        # 确保数据在GPU上
        if not isinstance(A, cp.ndarray):
            A_gpu = cp.asarray(A)
        else:
            A_gpu = A

        if not isinstance(B, cp.ndarray):
            B_gpu = cp.asarray(B)
        else:
            B_gpu = B

        # 分配输出内存
        C_gpu = cp.zeros((M, N), dtype=np.float32)

        # 计算网格和块的尺寸
        grid_dim = ((N + 15) // 16, (M + 15) // 16)
        block_dim = (16, 16)
        shared_mem = 2 * 16 * 16 * 4  # 2个16x16的浮点数块

        # 获取核心函数
        kernel = self.cuda_module.get_function("matrix_mul_kernel")

        # 启动核心
        kernel(grid_dim, block_dim,
               (A_gpu, B_gpu, C_gpu, M, N, K),
               shared_mem=shared_mem)

        return C_gpu

    def compute_ctc(self, C):
        """
        计算CT*C的优化实现
        C: 输入矩阵 [M, N]
        返回: result = C^T * C [N, N]
        """
        M, N = C.shape

        # 确保数据在GPU上
        if not isinstance(C, cp.ndarray):
            C_gpu = cp.asarray(C)
        else:
            C_gpu = C

        # 分配输出内存
        result_gpu = cp.zeros((N, N), dtype=np.float32)

        # 计算网格和块的尺寸
        grid_dim = ((N + 15) // 16, (N + 15) // 16)
        block_dim = (16, 16)
        shared_mem = 16 * 16 * 4  # 一个16x16的浮点数块

        # 获取核心函数
        kernel = self.cuda_module.get_function("ctc_kernel")

        # 启动核心
        kernel(grid_dim, block_dim,
               (C_gpu, result_gpu, M, N),
               shared_mem=shared_mem)

        return result_gpu

    def compute_residual(self, original, predicted):
        """
        计算残差的优化实现
        original: 原始数据 [size]
        predicted: 预测数据 [size]
        返回: residual = original - predicted [size]
        """
        size = len(original)

        # 确保数据在GPU上
        if not isinstance(original, cp.ndarray):
            original_gpu = cp.asarray(original)
        else:
            original_gpu = original

        if not isinstance(predicted, cp.ndarray):
            predicted_gpu = cp.asarray(predicted)
        else:
            predicted_gpu = predicted

        # 分配输出内存
        residual_gpu = cp.zeros(size, dtype=np.float32)

        # 计算网格和块的尺寸
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

        # 获取核心函数
        kernel = self.cuda_module.get_function("residual_kernel")

        # 启动核心
        kernel((blocks_per_grid,), (threads_per_block,),
               (original_gpu, predicted_gpu, residual_gpu, size))

        return residual_gpu

    def threshold_encode(self, residual, T_minus, T_plus):
        """
        双阈值编码的优化实现
        residual: 残差数据 [size]
        T_minus: 下阈值
        T_plus: 上阈值
        返回: 编码后的数据 [size]
        """
        size = len(residual)

        # 确保数据在GPU上
        if not isinstance(residual, cp.ndarray):
            residual_gpu = cp.asarray(residual)
        else:
            residual_gpu = residual

        # 分配输出内存
        encoded_gpu = cp.zeros(size, dtype=np.int32)

        # 计算网格和块的尺寸
        threads_per_block = 256
        blocks_per_grid = (size + threads_per_block - 1) // threads_per_block

        # 获取核心函数
        kernel = self.cuda_module.get_function("threshold_encode_kernel")

        # 启动核心
        kernel((blocks_per_grid,), (threads_per_block,),
               (residual_gpu, encoded_gpu, float(T_minus), float(T_plus), size))

        return encoded_gpu

    def __del__(self):
        """清理资源"""
        del self.cuda_module
        del self.streams


# 基于Mamba的DPCM压缩器
class MambaDPCMCompressor:
    def __init__(self, config=None):
        if config is None:
            self.config = MambaDPCMConfig()
        else:
            self.config = config

        # 创建各个组件
        self.dynamic_predictor = DynamicPredictionCoefficients(self.config)
        self.threshold_generator = AdaptiveThresholdGenerator(self.config)
        self.parallel_scan = ParallelScan(self.config)

        # 将模型移动到GPU
        self.dynamic_predictor = self.dynamic_predictor.to(device)
        self.threshold_generator = self.threshold_generator.to(device)

        # 初始化RangeCoder (自定义的范围编码器)
        self.range_coder = RangeCoder()

    def calculate_histogram(self, residual, bins=256):
        """计算残差直方图"""
        # 将残差值限制在合理范围内
        min_val = residual.min().item()
        max_val = residual.max().item()

        # 使用torch创建直方图
        hist = torch.histc(torch.tensor(residual, device=device),
                           bins=bins, min=min_val, max=max_val)

        # 归一化直方图
        hist = hist / hist.sum()

        return hist

    def compress(self, image):
        """
        压缩极光图像
        image: 输入图像 [height, width]
        返回: 压缩数据及元数据
        """
        start_time = time.time()

        # 确保图像是PyTorch张量
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32, device=device)

        # 获取图像尺寸
        height, width = image.shape

        # 分块处理图像
        block_size = self.config.block_size
        compressed_data = []
        metadata = {
            'height': height,
            'width': width,
            'block_size': block_size,
            'pred_order': self.config.pred_order,
            'thresholds': []
        }

        # 处理每个块
        for i in range(0, height, block_size):
            for j in range(0, width, block_size):
                # 提取当前块
                h_end = min(i + block_size, height)
                w_end = min(j + block_size, width)
                block = image[i:h_end, j:w_end]

                # 压缩当前块
                block_data = self._compress_block(block)
                compressed_data.append(block_data)

                # 记录当前块的阈值
                metadata['thresholds'].append((
                    block_data['T_minus'].item(),
                    block_data['T_plus'].item()
                ))

        end_time = time.time()
        print(f"压缩完成，耗时: {end_time - start_time:.4f}秒")

        return {
            'compressed_data': compressed_data,
            'metadata': metadata
        }

    def _compress_block(self, block):
        """
        压缩单个图像块
        block: 输入块 [block_height, block_width]
        返回: 压缩数据
        """
        height, width = block.shape

        # 预处理图像块为1D序列
        block_flat = block.reshape(-1)

        # 1. 使用动态预测系数生成模块预测像素值
        predicted_values = torch.zeros_like(block_flat)
        coefficients_list = []

        # 按行扫描进行预测
        pred_order = self.config.pred_order
        for i in range(len(block_flat)):
            if i < pred_order:
                # 前N个像素无法预测，直接存储
                predicted_values[i] = block_flat[i]
                continue

            # 获取历史像素
            history = block_flat[i - pred_order:i]

            # 生成动态预测系数
            with torch.no_grad():
                coeffs = self.dynamic_predictor(history.unsqueeze(0))
                coefficients_list.append(coeffs.squeeze().cpu().numpy())

            # 计算预测值
            pred_val = (coeffs.squeeze() * torch.flip(history, [0])).sum()
            predicted_values[i] = pred_val

        # 2. 计算残差
        residual = block_flat - predicted_values

        # 3. 计算残差直方图
        residual_hist = self.calculate_histogram(residual)

        # 4. 使用自适应阈值生成器确定阈值
        with torch.no_grad():
            T_minus, T_plus = self.threshold_generator(residual_hist.unsqueeze(0))

        # 5. 使用双阈值进行残差编码
        encoded_residual = torch.zeros_like(residual, dtype=torch.int32)
        outliers_mask = (residual < T_minus) | (residual > T_plus)
        outliers_indices = torch.nonzero(outliers_mask).squeeze()
        outliers_values = residual[outliers_mask]

        # 对正常范围内的残差使用RangeCoder编码
        normal_residual = residual[~outliers_mask].cpu().numpy()
        encoded_normal = self.range_coder.encode(normal_residual)

        # 将结果收集到一起
        return {
            'encoded_normal': encoded_normal,
            'outliers_indices': outliers_indices.cpu(),
            'outliers_values': outliers_values.cpu(),
            'T_minus': T_minus,
            'T_plus': T_plus,
            'coefficients': coefficients_list,
            'shape': (height, width)
        }

    # 接上文的MambaDPCMCompressor类中的decompress方法

    def decompress(self, compressed_data):
            """
            解压极光图像
            compressed_data: 压缩数据及元数据
            返回: 解压后的图像
            """
            metadata = compressed_data['metadata']
            blocks_data = compressed_data['compressed_data']

            # 创建输出图像
            height = metadata['height']
            width = metadata['width']
            block_size = metadata['block_size']
            decompressed_image = torch.zeros((height, width), device=device)

            # 处理每个块
            block_idx = 0
            for i in range(0, height, block_size):
                for j in range(0, width, block_size):
                    # 当前块的有效尺寸
                    h_end = min(i + block_size, height)
                    w_end = min(j + block_size, width)

                    # 解压当前块
                    block_data = blocks_data[block_idx]
                    decompressed_block = self._decompress_block(block_data)

                    # 将解压块放回原位置
                    decompressed_image[i:h_end, j:w_end] = decompressed_block

                    block_idx += 1

            return decompressed_image

    def _decompress_block(self, block_data):
            """
            解压单个图像块
            block_data: 单个块的压缩数据
            返回: 解压后的块
            """
            # 提取数据
            encoded_normal = block_data['encoded_normal']
            outliers_indices = block_data['outliers_indices']
            outliers_values = block_data['outliers_values']
            T_minus = block_data['T_minus']
            T_plus = block_data['T_plus']
            coefficients = block_data['coefficients']
            height, width = block_data['shape']

            # 创建输出块
            block_size = height * width
            reconstructed_block = torch.zeros(block_size, device=device)

            # 解码正常范围内的残差
            normal_residual = self.range_coder.decode(encoded_normal)

            # 创建完整的残差数组
            residual = torch.zeros(block_size, device=device)

            # 首先填充正常范围的残差
            normal_count = 0
            for i in range(block_size):
                if i not in outliers_indices:
                    if normal_count < len(normal_residual):
                        residual[i] = normal_residual[normal_count]
                        normal_count += 1

            # 然后填充异常值
            for idx, val in zip(outliers_indices, outliers_values):
                residual[idx] = val

            # 重建原始像素值
            pred_order = len(coefficients[0]) if coefficients else self.config.pred_order

            # 前N个像素直接复制
            for i in range(min(pred_order, block_size)):
                reconstructed_block[i] = residual[i]

            # 对其余像素应用预测和残差
            for i in range(pred_order, block_size):
                # 获取历史像素
                history = reconstructed_block[i - pred_order:i]

                # 使用对应位置的预测系数
                if i - pred_order < len(coefficients):
                    coeffs = torch.tensor(coefficients[i - pred_order], device=device)
                else:
                    # 容错处理
                    coeffs = torch.zeros(pred_order, device=device)

                # 计算预测值
                pred_val = (coeffs * torch.flip(history, [0])).sum()

                # 重建原始值
                reconstructed_block[i] = pred_val + residual[i]

            # 重新变形为2D块
            return reconstructed_block.reshape(height, width)

    # 范围编码器实现
    class RangeCoder:
        """
        范围编码器实现，用于残差的高效编码
        """

        def __init__(self, precision=16):
            self.precision = precision
            self.full_range = 1 << precision
            self.half_range = self.full_range >> 1
            self.quarter_range = self.full_range >> 2
            self.three_quarter_range = self.half_range + self.quarter_range

        def _estimate_frequencies(self, data):
            """估计数据频率分布"""
            # 数据类型转换
            data = np.array(data).astype(np.int32)

            # 找到唯一值及其计数
            values, counts = np.unique(data, return_counts=True)

            # 创建频率表
            freq_table = dict(zip(values, counts))

            # 计算累积频率
            total = sum(counts)
            cum_freq = {}
            cum_sum = 0

            for val in sorted(freq_table.keys()):
                cum_freq[val] = (cum_sum, cum_sum + freq_table[val])
                cum_sum += freq_table[val]

            return freq_table, cum_freq, total

        def encode(self, data):
            """
            对数据进行范围编码
            data: 输入数据 (numpy数组或列表)
            返回: 编码后的数据
            """
            if len(data) == 0:
                return {'encoded': [], 'freq_table': {}}

            # 估计频率
            freq_table, cum_freq, total = self._estimate_frequencies(data)

            # 初始化编码范围
            low = 0
            high = self.full_range - 1

            # 编码过程
            encoded_bits = []
            pending_bits = 0

            for symbol in data:
                # 更新范围
                range_size = high - low + 1
                low_cum, high_cum = cum_freq[symbol]

                high = low + (range_size * high_cum) // total - 1
                low = low + (range_size * low_cum) // total

                # 处理范围缩小
                while True:
                    if high < self.half_range:
                        # 输出0，增加pending 1s
                        encoded_bits.append(0)
                        for _ in range(pending_bits):
                            encoded_bits.append(1)
                        pending_bits = 0
                    elif low >= self.half_range:
                        # 输出1，增加pending 0s
                        encoded_bits.append(1)
                        for _ in range(pending_bits):
                            encoded_bits.append(0)
                        pending_bits = 0
                        low -= self.half_range
                        high -= self.half_range
                    elif low >= self.quarter_range and high < self.three_quarter_range:
                        # 缩小中间范围
                        pending_bits += 1
                        low -= self.quarter_range
                        high -= self.quarter_range
                    else:
                        break

                    # 左移
                    low <<= 1
                    high = (high << 1) | 1

                    # 确保范围在有效区间内
                    high &= self.full_range - 1
                    low &= self.full_range - 1

            # 输出最终位
            pending_bits += 1
            if low < self.quarter_range:
                encoded_bits.append(0)
                for _ in range(pending_bits):
                    encoded_bits.append(1)
            else:
                encoded_bits.append(1)
                for _ in range(pending_bits):
                    encoded_bits.append(0)

            # 将位流转换为字节
            encoded_bytes = []
            byte = 0
            bit_count = 0

            for bit in encoded_bits:
                byte = (byte << 1) | bit
                bit_count += 1
                if bit_count == 8:
                    encoded_bytes.append(byte)
                    byte = 0
                    bit_count = 0

            # 处理未满8位的尾部
            if bit_count > 0:
                byte <<= (8 - bit_count)
                encoded_bytes.append(byte)

            return {
                'encoded': encoded_bytes,
                'freq_table': freq_table
            }

        def decode(self, encoded_data):
            """
            解码数据
            encoded_data: 编码数据，包含编码和频率表
            返回: (解码后的数据)
            """
            encoded_bytes = encoded_data['encoded']
            freq_table = encoded_data['freq_table']

            # 如果输入为空，直接返回
            if not encoded_bytes:
                return []

            # 重建累积频率表
            values = sorted(freq_table.keys())
            counts = [freq_table[val] for val in values]
            total = sum(counts)

            cum_freq = {}
            cum_sum = 0
            for val, count in zip(values, counts):
                cum_freq[val] = (cum_sum, cum_sum + count)
                cum_sum += count

            # 准备解码
            # 将字节转换为位流
            encoded_bits = []
            for byte in encoded_bytes:
                for i in range(7, -1, -1):
                    bit = (byte >> i) & 1
                    encoded_bits.append(bit)

            # 初始化解码状态
            code = 0
            for i in range(self.precision):
                if i < len(encoded_bits):
                    code = (code << 1) | encoded_bits[i]

            low = 0
            high = self.full_range - 1

            # 解码过程
            decoded_data = []
            i = self.precision

            while True:
                # 查找当前码字对应的符号
                range_size = high - low + 1
                scaled_code = ((code - low + 1) * total - 1) // range_size

                symbol = None
                for val in values:
                    low_cum, high_cum = cum_freq[val]
                    if low_cum <= scaled_code < high_cum:
                        symbol = val
                        break

                if symbol is None:
                    break

                decoded_data.append(symbol)

                # 更新范围
                low_cum, high_cum = cum_freq[symbol]
                high = low + (range_size * high_cum) // total - 1
                low = low + (range_size * low_cum) // total

                # 读入新位并更新范围
                while True:
                    if high < self.half_range:
                        # 不需要调整
                        pass
                    elif low >= self.half_range:
                        # 调整范围
                        code -= self.half_range
                        low -= self.half_range
                        high -= self.half_range
                    elif low >= self.quarter_range and high < self.three_quarter_range:
                        # 缩小中间范围
                        code -= self.quarter_range
                        low -= self.quarter_range
                        high -= self.quarter_range
                    else:
                        break

                    # 左移并读入新位
                    low <<= 1
                    high = (high << 1) | 1
                    code = (code << 1)

                    if i < len(encoded_bits):
                        code |= encoded_bits[i]
                        i += 1

                    # 确保范围在有效区间内
                    high &= self.full_range - 1
                    low &= self.full_range - 1
                    code &= self.full_range - 1

                # 判断是否已经解码完毕
                if i >= len(encoded_bits) and low == 0 and high == self.full_range - 1:
                    break

            return decoded_data

    # 用于处理FITS格式的极光图像的工具类
    class AuroraImageProcessor:
        """
        用于处理FITS格式的极光图像
        """

        @staticmethod
        def load_fits_image(file_path):
            """
            加载FITS格式的极光图像
            file_path: FITS文件路径
            返回: 图像数据
            """
            with fits.open(file_path) as hdul:
                # 获取主数据
                data = hdul[0].data

                # FITS数据可能有多个维度，确保我们获取2D图像
                if data.ndim > 2:
                    data = data[0]  # 通常第一帧

                return data

        @staticmethod
        def save_fits_image(data, file_path, header=None):
            """
            保存图像为FITS格式
            data: 图像数据
            file_path: 输出文件路径
            header: 可选的FITS头信息
            """
            hdu = fits.PrimaryHDU(data)
            if header:
                hdu.header.update(header)

            hdul = fits.HDUList([hdu])
            hdul.writeto(file_path, overwrite=True)

        @staticmethod
        def preprocess_image(image):
            """
            预处理图像，标准化为适合模型的格式
            image: 输入图像
            返回: 预处理后的图像
            """
            # 确保数据类型正确
            image = image.astype(np.float32)

            # 处理异常值
            image = np.nan_to_num(image, nan=0.0, posinf=65535.0, neginf=0.0)

            # 确保值在合理范围内（16位图像）
            image = np.clip(image, 0, 65535)

            return image

    # 多GPU分配和协调的实现
    class MultiGPUManager:
        """
        管理多GPU的任务分配和协调
        """

        def __init__(self, config):
            self.config = config
            self.num_devices = min(config.devices, torch.cuda.device_count())

            if self.num_devices == 0:
                print("警告: 未检测到可用的GPU，将使用CPU")
                self.num_devices = 1

            print(f"使用 {self.num_devices} 个GPU设备")

            # 初始化每个设备上的流
            self.streams = []
            for i in range(self.num_devices):
                device_streams = []
                with torch.cuda.device(i):
                    for _ in range(config.streams_per_device):
                        device_streams.append(torch.cuda.Stream())
                self.streams.append(device_streams)

        def distribute_blocks(self, blocks):
            """
            将图像块分配给不同的GPU
            blocks: 图像块列表
            返回: GPU分配方案
            """
            # 简单的循环分配策略
            distribution = [[] for _ in range(self.num_devices)]
            for i, block in enumerate(blocks):
                device_id = i % self.num_devices
                distribution[device_id].append(block)

            return distribution

        def process_blocks(self, blocks, process_func):
            """
            使用多GPU并行处理图像块
            blocks: 图像块列表
            process_func: 处理函数，接收块和设备ID
            返回: 处理结果列表
            """
            # 分配块到不同GPU
            distribution = self.distribute_blocks(blocks)

            # 结果存储
            results = [None] * len(blocks)
            block_to_idx = {}
            idx = 0

            for device_id, device_blocks in enumerate(distribution):
                for block in device_blocks:
                    block_to_idx[id(block)] = idx
                    idx += 1

            # 并行处理
            events = []

            for device_id, device_blocks in enumerate(distribution):
                # 跳过空列表
                if not device_blocks:
                    continue

                # 将处理任务分配到不同流
                device_events = []
                num_streams = len(self.streams[device_id])

                for i, block in enumerate(device_blocks):
                    stream_id = i % num_streams
                    stream = self.streams[device_id][stream_id]

                    with torch.cuda.device(device_id), torch.cuda.stream(stream):
                        # 处理块
                        result = process_func(block, device_id)
                        results[block_to_idx[id(block)]] = result

                        # 记录事件
                        event = torch.cuda.Event()
                        event.record(stream)
                        device_events.append(event)

                events.extend(device_events)

            # 等待所有事件完成
            for event in events:
                event.synchronize()

            return results

    # 主函数：演示压缩流程
    def main():
        """
        主函数，演示完整的压缩和解压流程
        """
        print("基于Mamba的DPCM极光图像无损压缩器")

        # 配置参数
        config = MambaDPCMConfig(
            d_model=64,
            n_layer=4,
            d_state=16,
            block_size=256,
            pred_order=11,
            eq_count=7,
            threshold_init=13,
            streams_per_device=4,
            devices=torch.cuda.device_count()
        )

        # 创建压缩器
        compressor = MambaDPCMCompressor(config)

        # 示例：加载极光图像
        image_processor = AuroraImageProcessor()
        file_path = "aurora_image.fits"  # 替换为实际文件路径

        try:
            aurora_image = image_processor.load_fits_image(file_path)
        except Exception as e:
            print(f"加载图像时出错: {e}")
            print("使用随机生成的图像进行演示...")
            # 生成随机图像用于演示
            aurora_image = np.random.randint(0, 65536, (1024, 1024), dtype=np.uint16)

        # 预处理图像
        aurora_image = image_processor.preprocess_image(aurora_image)
        print(f"图像形状: {aurora_image.shape}")

        # 将图像转换为PyTorch张量
        image_tensor = torch.tensor(aurora_image, dtype=torch.float32, device=device)

        # 压缩图像
        print("开始压缩...")
        compressed_data = compressor.compress(image_tensor)

        # 计算压缩比
        original_size = aurora_image.nbytes
        compressed_size = sum(len(block['encoded_normal']['encoded']) for block in compressed_data['compressed_data'])
        compressed_size += sum(len(block['outliers_indices']) * 4 for block in compressed_data['compressed_data'])
        compressed_size += sum(len(block['outliers_values']) * 4 for block in compressed_data['compressed_data'])

        compression_ratio = original_size / max(compressed_size, 1)
        print(f"原始大小: {original_size} 字节")
        print(f"压缩后大小: {compressed_size} 字节")
        print(f"压缩比: {compression_ratio:.2f}:1")

        # 解压图像
        print("开始解压...")
        decompressed_image = compressor.decompress(compressed_data)

        # 计算PSNR (峰值信噪比)
        mse = ((image_tensor - decompressed_image) ** 2).mean().item()
        if mse > 0:
            psnr = 10 * math.log10((65535 ** 2) / mse)
        else:
            psnr = float('inf')  # 完全无损

        print(f"PSNR: {psnr:.2f} dB")

        # 验证无损性
        is_lossless = torch.allclose(image_tensor, decompressed_image)
        print(f"无损压缩验证: {'成功' if is_lossless else '失败'}")

        # 保存解压后的图像
        try:
            output_path = "decompressed_aurora.fits"
            image_processor.save_fits_image(
                decompressed_image.cpu().numpy(),
                output_path
            )
            print(f"解压图像已保存至: {output_path}")
        except Exception as e:
            print(f"保存图像时出错: {e}")

    # 入口点
    if __name__ == "__main__":
        main()
