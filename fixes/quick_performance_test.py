#!/usr/bin/env python3
"""
Medição rápida de performance - Detecção de Prateleiras Vazias
"""

import tensorflow as tf
import cv2
import numpy as np
import os
import time

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocessa uma imagem para o modelo"""
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_image, target_size)
    normalized = resized.astype(np.float32) / 255.0
    batch_input = np.expand_dims(normalized, axis=0)
    batch_input = np.expand_dims(batch_input, axis=-1)
    return batch_input

def quick_benchmark():
    """Executa benchmark rápido"""
    print("=== MEDIÇÃO RÁPIDA DE PERFORMANCE ===\n")
    
    # Carrega modelo
    model_path = "Models/empty_shelf_detector.h5"
    dataset_path = "Data/dataset/"
    
    print("Carregando modelo...")
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Modelo carregado!")
    
    # Carrega uma imagem de teste
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    test_image = os.path.join(dataset_path, image_files[0])
    
    print(f"Usando imagem: {image_files[0]}")
    
    # Warmup
    print("Executando warmup...")
    for _ in range(3):
        processed = preprocess_image(test_image)
        _ = model.predict(processed, verbose=0)
    
    # Medição
    print("Medindo performance...")
    times = []
    
    for i in range(10):
        start_time = time.perf_counter()
        processed = preprocess_image(test_image)
        bbox_pred, class_pred = model.predict(processed, verbose=0)
        end_time = time.perf_counter()
        
        inference_time = end_time - start_time
        times.append(inference_time)
        
        confidence = float(class_pred[0, 0])
        print(f"Teste {i+1}: {inference_time*1000:.2f}ms - Confiança: {confidence:.3f}")
    
    # Estatísticas
    avg_time = np.mean(times) * 1000  # ms
    min_time = np.min(times) * 1000   # ms
    max_time = np.max(times) * 1000   # ms
    fps = 1.0 / np.mean(times)
    
    print(f"\n📊 RESULTADOS:")
    print(f"   - Tempo médio: {avg_time:.2f} ms")
    print(f"   - Tempo mínimo: {min_time:.2f} ms")
    print(f"   - Tempo máximo: {max_time:.2f} ms")
    print(f"   - FPS médio: {fps:.2f}")
    print(f"   - Adequado para tempo real: {'✅ SIM' if fps >= 5 else '❌ NÃO'}")
    
    return {
        'avg_time_ms': avg_time,
        'min_time_ms': min_time,
        'max_time_ms': max_time,
        'fps': fps
    }

if __name__ == "__main__":
    quick_benchmark()
