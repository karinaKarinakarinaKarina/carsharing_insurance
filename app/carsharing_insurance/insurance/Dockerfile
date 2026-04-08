FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY insurance /app/insurance
COPY insurance/models /app/insurance/models

EXPOSE 5000

CMD ["python", "insurance/ml_service.py"]