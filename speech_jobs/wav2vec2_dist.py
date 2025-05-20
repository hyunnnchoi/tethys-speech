import tensorflow as tf
import numpy as np
import json
import os
import sys
import time
import argparse

parser = argparse.ArgumentParser(description='wav2vec2 Distributed Speech Recognition')
parser.add_argument('--num_batches', type=int, default=40, help='num_batches per replica, default is set 40')
parser.add_argument('--batch_size', type=int, default=1, help='batch size per replica, default is set 1')
args = parser.parse_args()

# 환경 설정
tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
task_config = tf_config.get('task', {})
task_type = task_config.get('type')
task_index = task_config.get('index')

# 모델과 프로세서를 저장할 로컬 디렉토리 설정
CACHE_DIR = '/workspace/model_cache'  # 컨테이너 내 사전 준비된 모델 캐시 경로
DATASET_DIR = '/workspace/datasets'  # 컨테이너 내 사전 준비된 데이터셋 경로

# 분산 학습 전략 설정
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# 하이퍼파라미터 설정
BATCH_SIZE_PER_REPLICA = args.batch_size
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
MAX_ITERATIONS = args.num_batches
BUFFER_SIZE = 10000

print(f'batch size per replica: {BATCH_SIZE_PER_REPLICA}, global batch size: {GLOBAL_BATCH_SIZE}')
print(f'num_batches: {MAX_ITERATIONS}')

# Wav2Vec2 모델을 흉내낸 간단한 TensorFlow 모델 정의
class SimpleWav2Vec2Model(tf.keras.Model):
    def __init__(self, num_layers=4, hidden_size=512):
        super(SimpleWav2Vec2Model, self).__init__()
        
        # Feature Extraction (CNN Layers)
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(80000,)),  # 5초 오디오 (16kHz)
            tf.keras.layers.Reshape((-1, 1)),  # Add channel dimension
            tf.keras.layers.Conv1D(64, kernel_size=10, strides=5, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(128, kernel_size=8, strides=4, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(256, kernel_size=4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(hidden_size, kernel_size=4, strides=2, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
        ])
        
        # Transformer Encoder Layers
        self.encoder_layers = []
        for _ in range(num_layers):
            self.encoder_layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.LayerNormalization(epsilon=1e-5),
                    # 중요: hidden_size를 512로 설정하여 차원 일치
                    tf.keras.layers.Bidirectional(
                        tf.keras.layers.LSTM(hidden_size // 2, return_sequences=True)
                    ),
                    tf.keras.layers.Dropout(0.1),
                ])
            )
        
        # Output Projection
        self.output_projection = tf.keras.layers.Dense(hidden_size)
        
    def call(self, inputs, training=False):
        # Feature Extraction
        hidden_states = self.feature_extractor(inputs)
        
        # Transformer Encoder
        for i, layer in enumerate(self.encoder_layers):
            # 잔차 연결 (residual connection)
            residual = hidden_states
            hidden_states = layer(hidden_states, training=training)
            hidden_states = hidden_states + residual
        
        # Output Projection
        outputs = self.output_projection(hidden_states)
        
        return {'last_hidden_state': outputs}

# 더미 오디오 데이터셋 생성
def create_dummy_dataset(batch_size):
    """
    학습용 더미 오디오 데이터셋을 생성합니다.
    """
    # 충분히 긴 더미 오디오 특성
    num_samples = 50  # 데이터셋 크기
    
    # 충분히 큰 더미 특성 생성 (16000Hz * 5초 = 80000 샘플)
    dummy_features = np.random.randn(num_samples, 80000).astype(np.float32)
    
    # 더미 레이블 (사용되지 않음)
    dummy_labels = np.zeros((num_samples, 1), dtype=np.float32)
    
    # TensorFlow 데이터셋 생성
    dataset = tf.data.Dataset.from_tensor_slices((dummy_features, dummy_labels))
    
    # 배치 설정 및 반복
    return dataset.batch(batch_size).repeat()

# 손실 함수 정의
def compute_loss(labels, outputs):
    """
    간단한 MSE 손실을 사용합니다.
    """
    # 출력의 일부 특성만 손실 계산에 사용
    loss = tf.reduce_mean(tf.square(outputs['last_hidden_state']))
    return loss

# 학습 스텝 정의
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        loss = compute_loss(labels, outputs)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# 분산 학습 스텝 정의
@tf.function
def distributed_train_step(inputs, labels):
    per_replica_losses = strategy.run(train_step, args=(inputs, labels))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# 메인 함수
def main():
    global model, optimizer
    
    print("Speech 분산 학습 시작...")
    
    with strategy.scope():
        # wav2vec2 모델 구조를 흉내낸 간단한 모델 초기화
        print("wav2vec2 스타일 모델 초기화 중...")
        model = SimpleWav2Vec2Model(hidden_size=512)  # 모든 차원을 512로 통일
        
        # 옵티마이저 설정
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    
    # 더미 데이터셋 생성
    print("더미 데이터셋 생성 중...")
    train_dataset = create_dummy_dataset(GLOBAL_BATCH_SIZE)
    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    # 학습 실행
    os.system('sh /workspace/network.sh &') # network profile
    os.system('sh /workspace/gpu.sh &') # gpu profile
    print('''
========================
network profile started!
========================''')
    # jct
    start_time = time.time()
    start_time_tf = tf.timestamp()

    iterator = iter(dist_dataset)
    total_loss = tf.constant(0.0)
    
    # 학습 루프
    for iteration in tf.range(MAX_ITERATIONS):
        inputs, labels = next(iterator)
        loss = distributed_train_step(inputs, labels)
        total_loss += loss

        tf.print("Timestamp: ", tf.timestamp() - start_time_tf, "Step:", iteration, "Loss:", loss, output_stream=sys.stdout)

    average_loss = total_loss / tf.cast(MAX_ITERATIONS, tf.float32)

    # jct
    end_time = time.time()
    jct = end_time - start_time

    # average_loss
    tf.print("Training completed. Average Loss:", average_loss, output_stream=sys.stdout)
    print("jct:", jct)
    
    # jct in tf_cnn_benchmark = wall_time in perfzero
    model_txt = open('/workspace/model.txt','r')
    save_dir_name = model_txt.read()
    jct_file = open('/result/' + save_dir_name.strip() + '/' + task_type + '_' + str(task_index) + '_jct.txt', 'w')
    jct_file.write('%.2f' % (float(jct)))
    jct_file.close()
    model_txt.close()

if __name__ == "__main__":
    main()