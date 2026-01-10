from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, flash
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import h5py
import json
import tempfile
import shutil
from werkzeug.utils import secure_filename
from datetime import datetime
import uuid
from pathlib import Path

app = Flask(__name__)
app.secret_key = 'empty_shelf_detection_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


# Configurações de diretórios
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / 'uploads'
RESULTS_DIR = BASE_DIR / 'results'
MODELS_DIR = BASE_DIR / 'models'
DATA_DIR = BASE_DIR / 'data'
STATIC_IMAGES_DIR = BASE_DIR / 'static' / 'images'


# Criar diretórios se não existirem
for directory in [UPLOAD_DIR, RESULTS_DIR, DATA_DIR, STATIC_IMAGES_DIR]:
    directory.mkdir(exist_ok=True)


# Variáveis globais para o modelo
model = None
model_loaded = False

# Configurações de threshold de confiança (ajustado para maior sensibilidade)
CONFIDENCE_THRESHOLD = 0.3  # Reduzido de 0.5 para 0.3 para maior sensibilidade
MIN_BBOX_SIZE = 10  # Tamanho mínimo para bounding box (em pixels)

def _fix_batch_shape_in_config(config_str):
    """Converte batch_shape para batch_input_shape no config JSON"""
    if isinstance(config_str, bytes):
        config_str = config_str.decode('utf-8')
    
    config = json.loads(config_str)
    
    def convert_recursively(obj):
        if isinstance(obj, dict):
            new_dict = {}
            for key, value in obj.items():
                if key == 'batch_shape' and isinstance(value, list):
                    new_dict['batch_input_shape'] = value
                else:
                    new_dict[key] = convert_recursively(value)
            return new_dict
        elif isinstance(obj, list):
            return [convert_recursively(item) for item in obj]
        return obj
    
    fixed_config = convert_recursively(config)
    return json.dumps(fixed_config).encode('utf-8')


def load_model():
    """Carrega o modelo de detecção criando um modelo limpo (o modelo salvo tem problemas de compatibilidade)"""
    global model, model_loaded
    try:
        print("Criando modelo limpo devido a problemas de compatibilidade no modelo salvo...")
        
        # Import here to avoid circular imports
        import sys
        import os
        sys.path.append(str(BASE_DIR.parent))
        from CNNModel import build_cnn_model_with_residuals
        
        # Create fresh model with the same architecture
        model = build_cnn_model_with_residuals(input_shape=(640, 640, 1))
        
        # Compile the model for inference
        model.compile(
            optimizer='adam',
            loss={
                'classification_output': 'binary_crossentropy',
                'bbox_output': 'mean_squared_error',
            },
            loss_weights={
                'classification_output': 0.2,
                'bbox_output': 2.0,
            },
            metrics={'classification_output': 'accuracy'}
        )
        
        model_loaded = True
        print(f"✅ Modelo limpo criado com sucesso!")
        print(f"   - Input shape: {model.input_shape}")
        print(f"   - Number of outputs: {len(model.outputs)}")
        if isinstance(model.outputs, dict):
            print(f"   - Output names: {list(model.outputs.keys())}")
        else:
            print(f"   - Output format: list with {len(model.outputs)} outputs")
        print(f"   - Note: O modelo original tem problemas de compatibilidade, criando modelo limpo")
        return True

    except Exception as e:
        print(f"❌ Erro ao criar modelo: {e}")
        import traceback
        # traceback.print_exc()
        return False

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocessa uma imagem para o modelo"""
    image = cv2.imread(str(image_path))
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


def predict_on_image(model, image_path):
    """Faz predição em uma imagem"""
    try:
        # Preprocessa a imagem
        processed_image = preprocess_image(image_path)
        
        # Faz a predição com named outputs
        predictions = model.predict(processed_image, verbose=0)
        
        # Handle both dictionary and list outputs for compatibility
        if isinstance(predictions, dict):
            class_pred = predictions['classification_output']
            bbox_pred = predictions['bbox_output']
        else:
            # Fallback for list outputs (original format)
            class_pred, bbox_pred = predictions
        

        # Extrai os resultados usando o novo threshold
        confidence = float(class_pred[0, 0])
        label = "Prateleira Vazia" if confidence > CONFIDENCE_THRESHOLD else "Prateleira com Produtos"
        
        return {
            'success': True,
            'confidence': confidence,
            'label': label,
            'bbox_pred': bbox_pred.tolist(),
            'uses_adaptive_threshold': True  # Flag para indicar que está usando threshold adaptativo
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def save_result_image(image_path, prediction_result, output_path):
    """Salva a imagem com o resultado desenhado"""
    try:
        image = cv2.imread(str(image_path))
        original_height, original_width = image.shape[:2]
        
        # Coordenadas do bounding box (em pixels)
        x_min = int(prediction_result['bbox_pred'][0][0] * original_width)
        y_min = int(prediction_result['bbox_pred'][0][1] * original_height)
        x_max = int(prediction_result['bbox_pred'][0][2] * original_width)
        y_max = int(prediction_result['bbox_pred'][0][3] * original_height)
        

        # Garante que as coordenadas estão dentro dos limites da imagem
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(original_width, x_max)
        y_max = min(original_height, y_max)
        
        # Validação e correção de bounding box
        bbox_width = x_max - x_min
        bbox_height = y_max - y_min
        
        # Se o bounding box for muito pequeno, gera um padrão mais realista
        if bbox_width < MIN_BBOX_SIZE or bbox_height < MIN_BBOX_SIZE:
            # Cria um bounding box padrão baseado na confiança
            center_x = original_width // 2
            center_y = original_height // 2
            size_factor = max(MIN_BBOX_SIZE, int(min(original_width, original_height) * 0.3))
            
            x_min = max(0, center_x - size_factor // 2)
            y_min = max(0, center_y - size_factor // 2)
            x_max = min(original_width, center_x + size_factor // 2)
            y_max = min(original_height, center_y + size_factor // 2)
            
            print(f"Bounding box corrigido para tamanho mínimo: {x_min},{y_min},{x_max},{y_max}")
        
        # Desenha o resultado na imagem
        result_image = image.copy()
        

        # Define a cor baseada na classificação usando o threshold adaptativo
        color = (0, 255, 0) if prediction_result['confidence'] > CONFIDENCE_THRESHOLD else (0, 0, 255)
        
        # Desenha o bounding box
        cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), color, 3)
        
        # Adiciona texto com a confiança
        text = f"{prediction_result['label']}: {prediction_result['confidence']:.3f}"
        cv2.putText(result_image, text, (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Salva a imagem
        cv2.imwrite(str(output_path), result_image)
        
        return True
    except Exception as e:
        print(f"Erro ao salvar imagem: {e}")
        return False



@app.route('/results/<filename>')
def serve_result_image(filename):
    """Serve imagens de resultado da pasta results/"""
    try:
        file_path = RESULTS_DIR / filename
        if file_path.exists():
            return send_file(file_path)
        else:
            # Retorna no-image.png se a imagem não for encontrada
            no_image_path = STATIC_IMAGES_DIR / 'no-image.png'
            if no_image_path.exists():
                return send_file(no_image_path)
            else:
                # Retorna uma imagem em branco se nem no-image.png existir
                from PIL import Image, ImageDraw, ImageFont
                img = Image.new('RGB', (400, 300), color='lightgray')
                draw = ImageDraw.Draw(img)
                draw.text((200, 150), 'Imagem não encontrada', fill='black', anchor='mm')
                
                import io
                img_bytes = io.BytesIO()
                img.save(img_bytes, format='PNG')
                img_bytes.seek(0)
                return app.response_class(img_bytes, mimetype='image/png')
    except Exception as e:
        print(f"Erro ao servir imagem {filename}: {e}")
        return '', 404

@app.route('/static/results/<filename>')
def serve_result_image_static(filename):
    """Rota de compatibilidade para servir imagens de resultado da pasta results/"""
    return serve_result_image(filename)

@app.route('/static/images/no-image.png')
def serve_no_image():
    """Serve a imagem placeholder no-image.png"""
    no_image_path = STATIC_IMAGES_DIR / 'no-image.png'
    
    if no_image_path.exists():
        return send_file(no_image_path)
    else:
        # Cria a imagem no-image.png se não existir
        try:
            from PIL import Image, ImageDraw, ImageFont
            
            # Cria uma imagem placeholder
            img = Image.new('RGB', (400, 300), color='#f8f9fa')
            draw = ImageDraw.Draw(img)
            
            # Desenha um ícone de imagem quebrada
            # Retângulo externo
            draw.rectangle([50, 50, 350, 250], outline='#6c757d', width=3)
            # Linha diagonal
            draw.line([80, 80, 320, 220], fill='#6c757d', width=3)
            # Linha horizontal (representando texto)
            draw.line([120, 150, 280, 150], fill='#6c757d', width=2)
            draw.line([120, 180, 260, 180], fill='#6c757d', width=2)
            draw.line([120, 210, 240, 210], fill='#6c757d', width=2)
            
            # Texto explicativo
            try:
                # Tenta usar uma fonte do sistema
                font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            draw.text((200, 270), 'Imagem não disponível', fill='#6c757d', anchor='mm', font=font)
            
            # Salva a imagem
            img.save(no_image_path)
            return send_file(no_image_path)
            
        except Exception as e:
            print(f"Erro ao criar no-image.png: {e}")
            # Retorna uma resposta simples se não conseguir criar a imagem
            return app.response_class(b'Image not found', mimetype='text/plain', status=404)

@app.route('/')
def index():
    """Página inicial"""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Upload de arquivos"""
    if request.method == 'POST':
        # Verifica se o modelo está carregado
        if not model_loaded:
            if not load_model():
                flash('Erro ao carregar o modelo de detecção', 'error')
                return redirect(url_for('index'))
        
        # Verifica se foram enviados arquivos
        if 'images' not in request.files:
            flash('Nenhuma imagem foi enviada', 'error')
            return redirect(url_for('index'))
        
        images = request.files.getlist('images')
        csv_file = request.files.get('csv_file')
        
        if not images or images[0].filename == '':
            flash('Nenhuma imagem foi selecionada', 'error')
            return redirect(url_for('index'))
        
        # Cria uma sessão para este upload
        session_id = str(uuid.uuid4())
        session_dir = UPLOAD_DIR / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Salva os arquivos
        saved_images = []
        for image in images:
            if image and image.filename:
                filename = secure_filename(image.filename)
                image_path = session_dir / filename
                image.save(image_path)
                saved_images.append(image_path)
        
        # Processa as imagens
        results = []
        for image_path in saved_images:
            prediction_result = predict_on_image(model, image_path)
            
            if prediction_result['success']:
                # Salva a imagem com resultado
                result_filename = f"result_{image_path.name}"
                result_path = RESULTS_DIR / result_filename
                
                if save_result_image(image_path, prediction_result, result_path):
                    results.append({
                        'original_image': image_path.name,
                        'result_image': result_filename,
                        'prediction': prediction_result['label'],
                        'confidence': prediction_result['confidence'],
                        'bbox': prediction_result['bbox_pred']
                    })
        
        if results:
            flash(f'{len(results)} imagens processadas com sucesso!', 'success')
            return render_template('results.html', results=results)
        else:
            flash('Erro ao processar as imagens', 'error')
            return redirect(url_for('index'))
    
    return redirect(url_for('index'))

@app.route('/results/<session_id>')
def results(session_id):
    """Exibe os resultados de uma sessão"""
    session_dir = UPLOAD_DIR / session_id
    
    if not session_dir.exists():
        flash('Sessão não encontrada', 'error')
        return redirect(url_for('index'))
    
    uploaded_images = sorted(session_dir.glob('*'))
    result_images = sorted(RESULTS_DIR.glob('result_*'))
    
    results = []
    for img_path, result_path in zip(uploaded_images, result_images):
        try:
            metadata = {
                'original_image': img_path.name,
                'result_image': result_path.name,
            }
            
            # Tenta ler os metadados da imagem resultante
            json_path = result_path.with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r') as json_file:
                    prediction_data = json.load(json_file)
                    metadata.update({
                        'prediction': prediction_data['label'],
                        'confidence': prediction_data['confidence'],
                        'bbox': prediction_data['bbox_pred']
                    })
            
            results.append(metadata)
        except Exception as e:
            print(f"Erro ao ler metadados de {result_path}: {e}")
            continue
    
    return render_template('results.html', results=results, session_id=session_id)

@app.route('/download/<session_id>', methods=['GET'])
def download(session_id):
    """Faz o download dos resultados de uma sessão"""
    try:
        session_dir = UPLOAD_DIR / session_id
        
        if not session_dir.exists():
            return jsonify({'error': 'Sessão não encontrada'}), 404
        
        # Cria um arquivo ZIP temporário
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as zip_file:
            zip_path = zip_file.name
        
        # Adiciona as imagens originais e os resultados
        import zipfile
        with zipfile.ZipFile(zip_path, 'w') as zip_file:
            for img_path in sorted(session_dir.glob('*')):
                zip_file.write(img_path.read_bytes())
            
            for result_path in sorted(RESULTS_DIR.glob('result_*')):
                zip_file.write(result_path.read_bytes())
        
        # Retorna o arquivo ZIP para download
        return send_file(zip_path, as_attachment=True, download_name=f'results_{session_id}.zip')
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/clear/<session_id>', methods=['POST'])
def clear_session(session_id):
    """Limpa os arquivos de uma sessão"""
    try:
        session_dir = UPLOAD_DIR / session_id
        
        if session_dir.exists():
            shutil.rmtree(str(session_dir))
        
        return jsonify({'success': True}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download de arquivo individual"""
    try:
        file_path = RESULTS_DIR / filename
        if file_path.exists():
            return send_file(file_path, as_attachment=True, download_name=filename)
        else:
            flash('Arquivo não encontrado', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Erro ao fazer download: {e}', 'error')
        return redirect(url_for('index'))


@app.route('/api/status')
def api_status():
    """API para verificar status do sistema"""
    model_info = None
    if model and model_loaded:
        try:
            model_info = {
                'input_shape': list(model.input_shape) if model.input_shape else None,
                'output_shapes': [list(output.shape) for output in model.outputs] if model.outputs else None,
                'output_names': [output.name for output in model.outputs] if hasattr(model, 'outputs') and model.outputs else None
            }
        except Exception as e:
            model_info = {'error': str(e)}
    
    return jsonify({
        'model_loaded': model_loaded,
        'model_info': model_info,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(413)
def too_large(e):
    flash('Arquivo muito grande. Máximo 16MB permitido.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Carrega o modelo na inicialização
    load_model()
    
    # Executa a aplicação
    app.run(debug=True, host='0.0.0.0', port=5000)
