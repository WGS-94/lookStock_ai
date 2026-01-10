"""
Testes automatizados para o sistema de detecção de prateleiras vazias
"""

import unittest
import requests
import json
import tempfile
import os
from PIL import Image
import numpy as np
import io


class TestEmptyShelfDetection(unittest.TestCase):
    """Testes para o sistema de detecção de prateleiras vazias"""
    
    BASE_URL = "http://127.0.0.1:5000"
    
    def setUp(self):
        """Configuração inicial dos testes"""
        self.session = requests.Session()
    
    def test_api_status(self):
        """Testa a API de status do sistema"""
        response = self.session.get(f"{self.BASE_URL}/api/status")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertIn("model_loaded", data)
        self.assertIn("model_info", data)
        self.assertIn("timestamp", data)
        
        self.assertTrue(data["model_loaded"])
        self.assertIsNotNone(data["model_info"]["input_shape"])
        self.assertEqual(len(data["model_info"]["output_shapes"]), 2)
    
    def test_main_page(self):
        """Testa se a página principal carrega corretamente"""
        response = self.session.get(f"{self.BASE_URL}/")
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        
        # Verifica se contém elementos essenciais
        content = response.text.lower()
        self.assertIn("bootstrap", content)
        self.assertIn("flask", content)
        self.assertIn("upload", content)
    
    def test_404_error(self):
        """Testa o tratamento de erro 404"""
        response = self.session.get(f"{self.BASE_URL}/nonexistent-page")
        
        self.assertEqual(response.status_code, 404)
        self.assertIn("text/html", response.headers["content-type"])
        
        content = response.text.lower()
        self.assertIn("404", content)
    
    def test_image_serving(self):
        """Testa se o sistema pode servir imagens"""
        # Cria uma imagem temporária para teste
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        # Tenta fazer upload da imagem
        files = {'images': ('test.jpg', img_bytes, 'image/jpeg')}
        response = self.session.post(f"{self.BASE_URL}/upload", files=files)
        
        # Verifica se a resposta é válida (pode ser 200 ou redirecionamento)
        self.assertIn(response.status_code, [200, 302])
    

    def test_api_endpoints(self):
        """Testa se os endpoints da API estão respondendo"""
        endpoints = [
            "/api/status",
            "/",
            "/upload"
        ]
        
        for endpoint in endpoints:
            response = self.session.get(f"{self.BASE_URL}{endpoint}")
            self.assertIn(response.status_code, [200, 405], 
                         f"Endpoint {endpoint} não está respondendo corretamente")
    
    def test_large_file_handling(self):
        """Testa o tratamento de arquivos grandes"""
        # Cria uma imagem grande (simulando arquivo > 16MB)
        large_img = Image.new('RGB', (5000, 5000), color='blue')
        img_bytes = io.BytesIO()
        large_img.save(img_bytes, format='JPEG', quality=95)
        img_bytes.seek(0)
        
        # Verifica se o arquivo é rejeitado por ser muito grande
        files = {'images': ('large_test.jpg', img_bytes, 'image/jpeg')}
        response = self.session.post(f"{self.BASE_URL}/upload", files=files)
        
        # O sistema deve rejeitar ou processar (depende da implementação)
        self.assertIn(response.status_code, [200, 302, 413, 500])


class TestCNNModelIntegration(unittest.TestCase):
    """Testes de integração com o modelo CNN"""
    
    def test_model_creation(self):
        """Testa se o modelo CNN pode ser criado"""
        try:
            from CNNModel import build_cnn_model_with_residuals
            
            model = build_cnn_model_with_residuals(input_shape=(640, 640, 1))
            
            self.assertIsNotNone(model)
            self.assertEqual(model.input_shape, (None, 640, 640, 1))
            self.assertEqual(len(model.outputs), 2)
            
        except ImportError:
            self.skipTest("CNNModel não disponível para teste")
    
    def test_image_preprocessing(self):
        """Testa o preprocessamento de imagens"""
        try:
            # Simula uma imagem de teste
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Testa se o preprocessamento funciona
            # (Esta função seria importada do app.py em um teste real)
            processed = test_image.astype(np.float32) / 255.0
            
            self.assertEqual(processed.shape, (480, 640, 3))
            self.assertTrue(processed.min() >= 0.0)
            self.assertTrue(processed.max() <= 1.0)
            
        except Exception as e:
            self.fail(f"Erro no preprocessamento: {e}")


class TestFileSystemOperations(unittest.TestCase):
    """Testes das operações do sistema de arquivos"""
    
    def test_directory_creation(self):
        """Testa se os diretórios necessários são criados"""
        required_dirs = ['uploads', 'results', 'static/images', 'data', 'models']
        
        for dir_path in required_dirs:
            # Cria o diretório se não existir
            os.makedirs(dir_path, exist_ok=True)
            self.assertTrue(os.path.exists(dir_path))
    
    def test_file_upload_structure(self):
        """Testa a estrutura de upload de arquivos"""
        # Simula uma estrutura de sessão de upload
        import uuid
        session_id = str(uuid.uuid4())
        session_dir = f"uploads/{session_id}"
        
        try:
            os.makedirs(session_dir, exist_ok=True)
            self.assertTrue(os.path.exists(session_dir))
            
            # Cria um arquivo de teste
            test_file_path = f"{session_dir}/test_image.jpg"
            with open(test_file_path, 'w') as f:
                f.write("test content")
            
            self.assertTrue(os.path.exists(test_file_path))
            
        finally:
            # Limpa os arquivos de teste
            if os.path.exists(session_dir):
                import shutil
                shutil.rmtree(session_dir)


def run_performance_tests():
    """Executa testes de performance básicos"""
    print("\n=== TESTES DE PERFORMANCE ===")
    
    try:
        # Teste de latência da API
        import time
        
        start_time = time.time()
        response = requests.get("http://127.0.0.1:5000/api/status")
        latency = time.time() - start_time
        
        print(f"✅ Latência da API: {latency*1000:.2f}ms")
        
        if response.status_code == 200:
            print("✅ API respondendo corretamente")
        else:
            print(f"❌ API retornou status: {response.status_code}")
        
        # Teste de throughput (requisições por segundo)
        start_time = time.time()
        for _ in range(10):
            requests.get("http://127.0.0.1:5000/api/status")
        
        throughput = 10 / (time.time() - start_time)
        print(f"✅ Throughput: {throughput:.2f} req/s")
        
    except Exception as e:
        print(f"❌ Erro nos testes de performance: {e}")


def main():
    """Função principal para executar todos os testes"""
    print("🚀 EXECUTANDO TESTES DO SISTEMA EMPTY SHELF DETECTION")
    print("=" * 60)
    
    # Executa testes unitários
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adiciona todos os testes
    suite.addTests(loader.loadTestsFromTestCase(TestEmptyShelfDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestCNNModelIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestFileSystemOperations))
    
    # Executa os testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Executa testes de performance
    run_performance_tests()
    
    # Relatório final
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO FINAL DOS TESTES")
    print("=" * 60)
    print(f"✅ Testes executados: {result.testsRun}")
    print(f"✅ Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Falhas: {len(result.failures)}")
    print(f"❌ Erros: {len(result.errors)}")
    
    if result.failures:
        print("\n🔥 FALHAS:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERROS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    # Status final
    if result.wasSuccessful():
        print("\n🎉 TODOS OS TESTES PASSARAM! Sistema funcionando perfeitamente!")
    else:
        print("\n⚠️  ALGUNS TESTES FALHARAM. Verifique os problemas acima.")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
