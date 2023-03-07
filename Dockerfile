# Build Python libraries
FROM python:3.8-buster as python-build
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt EXPOSE 5000 CMD streamlit run src/app/main.py
