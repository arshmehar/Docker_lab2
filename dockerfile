#only installing these dependencies once(took twice as long)

FROM python:3.10 AS base
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt


# Stage 1: Train the model
FROM base AS model_training
COPY data/ /app/data/                
COPY src/model_training.py /app/
RUN python model_training.py          

# Stage 2: Serve predictions
FROM base AS serving
COPY --from=model_training /app/my_model.keras /app/
COPY --from=model_training /app/scaler.pkl /app/    
COPY src/main.py /app/
COPY src/templates /app/templates

#flask runs on 4000 not 80
EXPOSE 4000                           
CMD ["python", "main.py"]
