# FROM python:3.9
# WORKDIR /app
# COPY requirements.txt ./requirements.txt
# RUN pip install -r requirements.txt
# EXPOSE 8501
# COPY . /app
# ENTRYPOINT ["streamlit", "run"]
# CMD ["main.py"]


FROM python:3.11-slim-buster

LABEL Name="Python Demo App" Version=1.0.0

ARG srcDir=src
WORKDIR /app
COPY $srcDir/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY $srcDir/run.py .
COPY $srcDir/app ./app
# Expose port you want your app on
EXPOSE 8080

# Run
ENTRYPOINT [“streamlit”, “run”, testchat2.py”, “–server.port=8080”, “–server.address=0.0.0.0”]
