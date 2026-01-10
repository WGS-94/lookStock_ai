#!/usr/bin/env python3
"""
Benchmark de performance do sistema de detecção de prateleiras vazias
Mede FPS, latência e throughput do modelo
"""

import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime
from tensorflow.keras import layers, Input
import statistics

def build_cnn_model_with_residuals(input_shape=(640, 640, 1)):
    """Reconstrói o modelo CNN com conexões residuais"""
    input_tensor = Input(shape=input_shape)
    
    x = layers.Conv2D(32, (7, 7), activation='relu', padding='same')(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    shortcut = x
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    shortcut = layers.Conv2D(64, (1, 1), strides=(2, 2), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    
    shortcut = x
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    shortcut = layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    
    shortcut = x
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    shortcut = layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    
    shortcut = x
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    shortcut = layers.Conv2D(512, (1, 1), strides=(2, 2), padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    classification_output = layers.Dense(1, activation='sigmoid', name='classification_output')(x)
    bbox_output = layers.Dense(4, name='bbox_output')(x)
    
    model = tf.keras.Model(inputs=input_tensor, outputs=[bbox_output, classification_output])
    
    return model

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocessa uma imagem para o modelo"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
    
    # Converte para escala de cinza (1 canal) como esperado pelo modelo
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Redimensiona para o tamanho esperado pelo modelo
    resized = cv2.resize(gray_image, target_size)
    
    # Normaliza para [0,1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Adiciona dimensões do batch e canal
    batch_input = np.expand_dims(normalized, axis=0)
    batch_input = np.expand_dims(batch_input, axis=-1)  # Forma final: (1, 640, 640, 1)
    
    return batch_input

def single_inference(model, image_path):
    """Executa uma única inferência e retorna o tempo"""
    start_time = time.perf_counter()
    
    try:
        # Preprocessa a imagem
        processed_image = preprocess_image(image_path)
        
        # Faz a predição
        bbox_pred, class_pred = model.predict(processed_image, verbose=0)
        
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        
        confidence = float(class_pred[0, 0])
        label = "Prateleira Vazia" if confidence > 0.5 else "Prateleira com Produtos"
        
        return {
            'success': True,
            'inference_time': inference_time,
            'confidence': confidence,
            'label': label
        }
        
    except Exception as e:
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        
        return {
            'success': False,
            'inference_time': inference_time,
            'error': str(e)
        }

def benchmark_model(model, image_paths, num_warmup=5, num_runs=20):
    """Executa benchmark completo do modelo"""
    print("🔥 Iniciando benchmark de performance...\n")
    
    # Warmup runs (não contam para a média)
    print(f"1. Executando {num_warmup} execuções de warmup...")
    for i in range(num_warmup):
        test_image = image_paths[i % len(image_paths)]
        single_inference(model, test_image)
    
    print(f"✅ Warmup concluído!\n")
    
    # Benchmark runs (contam para a média)
    print(f"2. Executando {num_runs} inferências para benchmark...")
    inference_times = []
    successful_runs = 0
    
    for i in range(num_runs):
        test_image = image_paths[i % len(image_paths)]
        result = single_inference(model, test_image)
        
        if result['success']:
            inference_times.append(result['inference_time'])
            successful_runs += 1
            
            if (i + 1) % 5 == 0:
                current_fps = 1.0 / statistics.mean(inference_times[-5:])
                print(f"   Progresso: {i+1}/{num_runs} - FPS atual: {current_fps:.2f}")
    
    if len(inference_times) == 0:
        print("❌ Nenhuma inferência foi bem-sucedida!")
        return None
    
    # Calcula estatísticas
    avg_inference_time = statistics.mean(inference_times)
    median_inference_time = statistics.median(inference_times)
    min_inference_time = min(inference_times)
    max_inference_time = max(inference_times)
    std_inference_time = statistics.stdev(inference_times) if len(inference_times) > 1 else 0
    
    fps_avg = 1.0 / avg_inference_time
    fps_median = 1.0 / median_inference_time
    fps_max = 1.0 / min_inference_time
    
    return {
        'successful_runs': successful_runs,
        'total_runs': num_runs,
        'avg_inference_time': avg_inference_time,
        'median_inference_time': median_inference_time,
        'min_inference_time': min_inference_time,
        'max_inference_time': max_inference_time,
        'std_inference_time': std_inference_time,
        'fps_avg': fps_avg,
        'fps_median': fps_median,
        'fps_max': fps_max,
        'all_times': inference_times
    }

def main():
    """Função principal de benchmark"""
    print("=== BENCHMARK DE PERFORMANCE - DETECÇÃO DE PRATELEIRAS VAZIAS ===\n")
    
    # Configurações
    model_path = "Models/empty_shelf_detector.h5"
    dataset_path = "Data/dataset/"
    
    # Verifica arquivos
    print("📁 Verificando arquivos...")
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return
    
    print("✅ Arquivos encontrados!\n")
    
    # Carrega modelo
    print("🤖 Carregando modelo...")
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✅ Modelo carregado com sucesso!")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Output shapes: {[output.shape for output in model.outputs]}")
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return
    

    # Carrega imagens de teste
    print(f"\n📸 Preparando imagens de teste...")
    image_files = [f for f in os.listdir(dataset_path) if f.endswith('.jpg')]
    
    if len(image_files) < 3:
        print(f"❌ Poucas imagens para benchmark. Encontradas: {len(image_files)}")
        return
    
    # Usa todas as imagens disponíveis
    num_images = min(len(image_files), 15)  # Máximo 15 imagens
    image_paths = [os.path.join(dataset_path, f) for f in image_files[:num_images]]
    print(f"✅ {len(image_paths)} imagens selecionadas para teste\n")
    
    # Executa benchmark
    results = benchmark_model(model, image_paths, num_warmup=5, num_runs=30)
    
    if results is None:
        return
    
    # Exibe resultados
    print("\n" + "="*60)
    print("📊 RESULTADOS DO BENCHMARK")
    print("="*60)
    
    print(f"\n🔢 ESTATÍSTICAS DE EXECUÇÃO:")
    print(f"   - Execuções bem-sucedidas: {results['successful_runs']}/{results['total_runs']}")
    print(f"   - Taxa de sucesso: {(results['successful_runs']/results['total_runs']*100):.1f}%")
    
    print(f"\n⏱️  TEMPO DE PROCESSAMENTO POR IMAGEM:")
    print(f"   - Tempo médio: {results['avg_inference_time']*1000:.2f} ms")
    print(f"   - Tempo mediano: {results['median_inference_time']*1000:.2f} ms")
    print(f"   - Tempo mínimo: {results['min_inference_time']*1000:.2f} ms")
    print(f"   - Tempo máximo: {results['max_inference_time']*1000:.2f} ms")
    print(f"   - Desvio padrão: {results['std_inference_time']*1000:.2f} ms")
    
    print(f"\n🚀 VELOCIDADE DE PROCESSAMENTO:")
    print(f"   - FPS médio: {results['fps_avg']:.2f}")
    print(f"   - FPS mediano: {results['fps_median']:.2f}")
    print(f"   - FPS máximo: {results['fps_max']:.2f}")
    
    print(f"\n💡 INTERPRETAÇÃO:")
    if results['fps_avg'] >= 10:
        performance_level = "EXCELENTE"
    elif results['fps_avg'] >= 5:
        performance_level = "BOA"
    elif results['fps_avg'] >= 1:
        performance_level = "ACEITÁVEL"
    else:
        performance_level = "LENTA"
    
    print(f"   - Performance geral: {performance_level}")
    print(f"   - Adequado para aplicação em tempo real: {'✅ SIM' if results['fps_avg'] >= 5 else '❌ NÃO'}")
    
    # Salva relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"Results/performance_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RELATÓRIO DE PERFORMANCE - DETECÇÃO DE PRATELEIRAS VAZIAS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Modelo: {model_path}\n")
        f.write(f"Arquitetura: CNN personalizada com conexões residuais\n")
        f.write(f"Framework: TensorFlow/Keras\n")
        f.write(f"Input: 640x640x1 (escala de cinza)\n\n")
        
        f.write("RESULTADOS:\n")
        f.write(f"- Execuções bem-sucedidas: {results['successful_runs']}/{results['total_runs']}\n")
        f.write(f"- Tempo médio por imagem: {results['avg_inference_time']*1000:.2f} ms\n")
        f.write(f"- FPS médio: {results['fps_avg']:.2f}\n")
        f.write(f"- Performance: {performance_level}\n")
    
    print(f"\n📄 Relatório salvo em: {report_path}")
    print("\n=== BENCHMARK CONCLUÍDO ===")

if __name__ == "__main__":
    main()
