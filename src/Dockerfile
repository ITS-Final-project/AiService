FROM python

WORKDIR /app

COPY . ./

RUN pip install numpy python-socketio asyncio flask[async] opencv-python-headless matplotlib imageio requests python-dotenv

CMD ["python", "main.py"]
