FROM python:3.8.19
WORKDIR /CI_ML
COPY . /CI_ML
EXPOSE 5000
RUN pip install -r requirements.txt 
CMD ["python","app.py"]