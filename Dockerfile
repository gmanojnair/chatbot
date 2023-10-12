# FROM python:3.9
# WORKDIR /app
# COPY requirements.txt ./requirements.txt
# RUN pip install -r requirements.txt
# EXPOSE 8501
# COPY . /app
# ENTRYPOINT ["streamlit", "run"]
# CMD ["main.py"]

FROM python:3.11.6

# Expose port you want your app on
EXPOSE 8080

# Upgrade pip and install requirements
COPY requirement.txt requirement.txt
RUN pip install -U pip
RUN pip install -r requirement.txt

# Copy app code and set working directory
COPY . .
WORKDIR /app

# Run
ENTRYPOINT [“streamlit”, “run”, testchat2.py”, “–server.port=8080”, “–server.address=0.0.0.0”]