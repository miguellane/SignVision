FROM python:3.10
WORKDIR /app
COPY ./recogrequirements.txt ./recogrequirements.txt
RUN pip install -r recogrequirements.txt
COPY . .
#CMD ["python", "-m", "recog", "run", "--host=0.0.0.0"]
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]