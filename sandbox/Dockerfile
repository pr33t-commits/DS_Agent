FROM python:3.12-slim
RUN pip install --no-cache-dir pandas==2.2.3 numpy==2.2.6 scipy==1.15.3
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
USER 65534:65534
WORKDIR /tmp
ENTRYPOINT ["python", "-I", "-B", "-"]
