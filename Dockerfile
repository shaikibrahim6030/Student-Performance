FROM python:3.10
EXPOSE 5000
WORKDIR /opt/app
COPY . /opt/app
RUN pip3 install -r requirements.txt
RUN python3 train.py
CMD ["python3", "/opt/app/app.py"]
