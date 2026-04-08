FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt /app/

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . /app/

EXPOSE 7000

CMD ["python", "manage.py", "runserver", "0.0.0.0:7000"]