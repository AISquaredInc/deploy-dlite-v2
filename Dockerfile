FROM python:3.10

RUN apt update
RUN apt-get install -y python3 python3-pip

ADD main.py .
ADD requirements.txt .

EXPOSE 2244

RUN pip install -r requirements.txt

CMD ["python", "./main.py"]