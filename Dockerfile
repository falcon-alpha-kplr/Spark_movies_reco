FROM apache/spark:3.4.0
WORKDIR /app
COPY ./requirements.txt  /app/requirements.txt
COPY ./app  /app
RUN pip install -r requirements.txt
COPY  . .
EXPOSE 5432
CMD ["spark-submit", "server.py", "ml-latest/movies.csv", "ml-latest/ratings.csv"]