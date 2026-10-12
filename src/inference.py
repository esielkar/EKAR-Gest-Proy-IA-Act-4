import time
import torch
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from preprocess import load_and_prepare_data

def evaluate_models():
    device = 0 if torch.cuda.is_available() else -1
    print(f"Ejecutando en dispositivo GPU ID: {device}")
    
    dataset = load_and_prepare_data()
    
    model_ids = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "pysentimiento/robertuito-sentiment-analysis",
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ]
    
    results = []
    
    for model_id in model_ids:
        print(f"\nEvaluando modelo: {model_id}")
        classifier = pipeline("text-classification", model=model_id, device=device)
        
        start_time = time.time()
        predictions = []
        for text in dataset["text"]:
            pred = classifier(text, truncation=True, max_length=128)
            label_res = pred[0]['label'].upper()
            is_pos = any(k in label_res for k in ['POS', 'POSITIVE', 'LABEL_2'])
            predictions.append(1 if is_pos else 0)
            
        end_time = time.time()
        latency = ((end_time - start_time) * 1000) / len(dataset)
        
        y_true = dataset["label"]
        y_pred = [1 if p == 1 else 0 for p in predictions]
        
        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        results.append({
            "Modelo": model_id,
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Latencia (ms)": round(latency, 2)
        })
        
    df_results = pd.DataFrame(results)
    df_results.to_csv("results/evaluation_metrics.csv", index=False)
    print("\nResultados guardados en results/evaluation_metrics.csv")
    print(df_results)

if __name__ == "__main__":
    evaluate_models()
