FROM python:3.10
ADD main.py .
COPY . .
RUN pip3 install numpy
CMD ["python3", "main.py"]

