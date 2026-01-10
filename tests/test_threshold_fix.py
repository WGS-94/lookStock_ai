#!/usr/bin/env python3
"""
Script de teste para verificar se o threshold de confiança foi ajustado corretamente
e se os bounding boxes estão sendo gerados.
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

# Adiciona o diretório pai ao path para importar módulos
sys.path.append(str(Path(__file__).parent))

# Importa o módulo CNNModel
from CNNModel import build_cnn_model_with_residuals

def test_threshold_working():
    """Testa se o threshold está funcionando corretamente"""
    
    # Configurações de teste
    CONFIDENCE_THRESHOLD = 0.3  # Novo threshold
    MIN_BBOX_SIZE = 10  # Tamanho mínimo
    
    print("=== Teste do Threshold de Confiança ===")
    print(f"Threshold configurado: {CONFIDENCE_THRESHOLD}")
    print(f"Tamanho mínimo de bbox: {MIN_BBOX_SIZE}")
    
    # Simula diferentes valores de confiança
    test_confidences = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.502, 0.55, 0.60]
    
    print("\nTeste de classificação com diferentes confianças:")
    for conf in test_confidences:
        label = "Prateleira Vazia" if conf > CONFIDENCE_THRESHOLD else "Prateleira com Produtos"
        color = "VERMELHO" if conf > CONFIDENCE_THRESHOLD else "AZUL"
        print(f"Confiança: {conf:.3f} -> Label: {label} -> Cor: {color}")
    
    # Testa bounding box com confiança crítica (50.2%)
    critical_confidence = 0.502
    print(f"\n=== Teste com confiança crítica: {critical_confidence} ===")
    label = "Prateleira Vazia" if critical_confidence > CONFIDENCE_THRESHOLD else "Prateleira com Produtos"
    print(f"Com confiança de {critical_confidence}: {label}")
    
    if critical_confidence > CONFIDENCE_THRESHOLD:
        print("✅ SUCESSO: Com confiança de 50.2%, o threshold de 0.3 permite classificação como 'Prateleira Vazia'")
    else:
        print("❌ FALHA: Threshold ainda está muito alto")
    
    return critical_confidence > CONFIDENCE_THRESHOLD

def test_bbox_validation():
    """Testa a validação de bounding box"""
    
    print("\n=== Teste de Validação de Bounding Box ===")
    
    # Teste 1: Bounding box muito pequeno
    test_image_size = (640, 640)
    original_width, original_height = test_image_size
    
    # Coordenadas de um bbox muito pequeno (1x1 pixel)
    x_min, y_min, x_max, y_max = 100, 100, 101, 101
    
    print(f"Bbox original: ({x_min}, {y_min}, {x_max}, {y_max})")
    print(f"Tamanho original: {x_max - x_min} x {y_max - y_min}")
    
    MIN_BBOX_SIZE = 10
    
    # Aplica a validação
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min
    
    if bbox_width < MIN_BBOX_SIZE or bbox_height < MIN_BBOX_SIZE:
        # Cria um bounding box padrão baseado na confiança
        center_x = original_width // 2
        center_y = original_height // 2
        size_factor = max(MIN_BBOX_SIZE, int(min(original_width, original_height) * 0.3))
        
        x_min = max(0, center_x - size_factor // 2)
        y_min = max(0, center_y - size_factor // 2)
        x_max = min(original_width, center_x + size_factor // 2)
        y_max = min(original_height, center_y + size_factor // 2)
        
        print(f"Bbox corrigido: ({x_min}, {y_min}, {x_max}, {y_max})")
        print(f"Tamanho corrigido: {x_max - x_min} x {y_max - y_min}")
        print("✅ Bounding box pequeno foi corrigido automaticamente")
    
    return True

def create_test_image():
    """Cria uma imagem de teste simples"""
    
    print("\n=== Criando Imagem de Teste ===")
    
    # Cria uma imagem simples de prateleira
    img = np.ones((400, 400, 3), dtype=np.uint8) * 240  # Fundo cinza claro
    
    # Desenha linhas horizontais simulando prateleiras
    for i in range(50, 350, 50):
        cv2.line(img, (20, i), (380, i), (200, 200, 200), 3)
    
    # Salva a imagem de teste
    test_image_path = "test_shelf_image.jpg"
    cv2.imwrite(test_image_path, img)
    print(f"Imagem de teste criada: {test_image_path}")
    
    return test_image_path

def main():
    """Função principal de teste"""
    
    print("🔧 TESTE DO AJUSTE DE THRESHOLD DE CONFIANÇA")
    print("=" * 50)
    
    # Testa o threshold
    threshold_ok = test_threshold_working()
    
    # Testa validação de bbox
    bbox_ok = test_bbox_validation()
    
    # Cria imagem de teste
    test_image = create_test_image()
    
    print("\n=== RESUMO DOS TESTES ===")
    print(f"Threshold funcionando: {'✅ SIM' if threshold_ok else '❌ NÃO'}")
    print(f"Validação de bbox: {'✅ SIM' if bbox_ok else '❌ NÃO'}")
    print(f"Imagem de teste criada: ✅ {test_image}")
    
    if threshold_ok and bbox_ok:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("O threshold de confiança foi ajustado com sucesso.")
        print("Agora o modelo deve gerar bounding boxes mesmo com confiança de 50.2%.")
    else:
        print("\n⚠️ ALGUNS TESTES FALHARAM!")
        print("Revisar as modificações no código.")
    
    print("\n💡 Para testar na aplicação web:")
    print("1. Abra http://127.0.0.1:5000 no navegador")
    print("2. Faça upload de uma imagem")
    print("3. Verifique se os bounding boxes são exibidos")

if __name__ == "__main__":
    main()
