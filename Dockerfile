FROM python:3.9-slim



WORKDIR /sahilfirsthack



ENV HOST 0.0.0.0



COPY requirements.txt .



RUN pip install -r requirements.txt



COPY . .


EXPOSE 8000


CMD ["uvicorn", "fastgpt:app", "--host", "0.0.0.0", "--port", "8000"]





