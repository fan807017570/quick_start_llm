# =============================================================================
# 新实中学招生顾问 RAG 服务 Dockerfile
# 构建上下文：quick_start_llm/ 目录
#   docker build -t xinshi-rag -f arg/xinshi/Dockerfile .
# =============================================================================

FROM python:3.11-slim

# 防止交互式安装提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HuggingFace 缓存目录（挂载宿主机目录可避免每次重新下载模型）
ENV HF_HOME=/app/hf_cache
ENV TRANSFORMERS_CACHE=/app/hf_cache

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 层缓存
COPY arg/xinshi/requirements.txt /app/requirements.txt

# 安装 Python 依赖
# torch CPU-only wheel（Linux/amd64，减小镜像体积约 1 GB）
RUN pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.6.0+cpu" && \
    pip install --no-cache-dir -r requirements.txt

# 复制完整项目（保持包结构 arg/xinshi/...）
COPY arg/ /app/arg/

# 确保包可被 Python 发现
ENV PYTHONPATH=/app

# 服务端口
EXPOSE 8765

# 启动命令（支持通过 XINSHI_HOST / XINSHI_PORT 环境变量覆盖）
CMD ["python", "-m", "uvicorn", "arg.xinshi.server:app", "--host", "0.0.0.0", "--port", "8765"]
