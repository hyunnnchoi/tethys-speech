import tensorflow as tf
import numpy as np
import json
import os
import time
import argparse
from tensorflow.keras import layers, Model
from scipy import stats  # skewness 계산을 위해 추가


# 텐서 사이즈 측정을 위한 전역 변수들
TENSOR_SIZE_TRACKER = {
    'current_step_tensors': [],
    'total_tensor_size': 0,
    'step_tensor_sizes': [],
    'operation_tensor_sizes': {}
}


class TensorProfiler:
    """Tiresias 논문과 같은 방식으로 텐서 사이즈를 측정하는 고급 프로파일러"""
    
    def __init__(self, log_dir='/workspace/tensor_logs'):
        self.log_dir = log_dir
        self.current_step = 0
        self.current_step_size = 0
        self.step_tensor_sizes = []
        self.operation_tensor_sizes = {}
        self.tensor_details = []
        self.gradient_sizes = []
        self.parameter_sizes = []
        self.memory_usage = []
        
        # 로그 파일 초기화
        os.makedirs(log_dir, exist_ok=True)
        
        # 텐서 사이즈 로그 파일
        self.tensor_log_file = open(os.path.join(log_dir, 'tensor_sizes.txt'), 'w')
        self.tensor_log_file.write("step,operation,tensor_type,size_bytes,size_mb,shape\n")
        
        # 메모리 사용량 로그 파일  
        self.memory_log_file = open(os.path.join(log_dir, 'memory_usage.txt'), 'w')
        self.memory_log_file.write("step,gpu_memory_mb,cpu_memory_mb\n")
        
        # 요약 로그 파일
        self.summary_log_file = open(os.path.join(log_dir, 'summary.txt'), 'w')
        self.summary_log_file.write("step,total_tensor_size_mb,num_operations,avg_tensor_size_mb\n")
        
        # Tiresias 스타일 텐서 사이즈 로그 파일
        self.tiresias_log_file = open(os.path.join(log_dir, 'tiresias_tensorsize.txt'), 'w')
        self.tiresias_log_file.write("step,tensorsize_mb\n")
        
        print(f"🔍 TensorProfiler 초기화됨 - 로그 디렉토리: {log_dir}")
    
    def log_tensor_size(self, tensor, name, tensor_type="activation"):
        """텐서 사이즈를 로깅"""
        if tensor is None:
            return 0
            
        try:
            # 텐서 크기 계산
            size_bytes = self._calculate_tensor_size(tensor)
            size_mb = size_bytes / (1024 * 1024)
            
            # 형태 정보 가져오기
            try:
                shape = tensor.shape.as_list() if hasattr(tensor.shape, 'as_list') else list(tensor.shape)
            except:
                shape = "unknown"
            
            # 현재 스텝 크기에 추가
            self.current_step_size += size_bytes
            
            # Operation별 크기 추적
            if name not in self.operation_tensor_sizes:
                self.operation_tensor_sizes[name] = []
            self.operation_tensor_sizes[name].append(size_bytes)
            
            # 상세 정보 저장
            tensor_info = {
                'step': self.current_step,
                'operation': name,
                'tensor_type': tensor_type,
                'size_bytes': size_bytes,
                'size_mb': size_mb,
                'shape': shape
            }
            self.tensor_details.append(tensor_info)
            
            # 파일에 즉시 로깅 (메모리 절약)
            self.tensor_log_file.write(f"{self.current_step},{name},{tensor_type},{size_bytes},{size_mb:.4f},{shape}\n")
            self.tensor_log_file.flush()
            
            return size_bytes
            
        except Exception as e:
            print(f"텐서 사이즈 로깅 오류: {e}")
            return 0
    
    def log_gradients(self, gradients, variables):
        """그래디언트 텐서들의 사이즈를 로깅"""
        for i, (grad, var) in enumerate(zip(gradients, variables)):
            if grad is not None:
                var_name = getattr(var, 'name', f'variable_{i}')
                self.log_tensor_size(grad, f"gradient_{var_name}", "gradient")
    
    def log_model_parameters(self, model):
        """모델 파라미터들의 사이즈를 로깅"""
        total_params = 0
        trainable_params = 0
        
        for var in model.trainable_variables:
            param_size = self.log_tensor_size(var, f"param_{var.name}", "parameter")
            total_params += param_size
            trainable_params += param_size
        
        for var in model.non_trainable_variables:
            param_size = self.log_tensor_size(var, f"param_{var.name}", "parameter")
            total_params += param_size
        
        # 파라미터 통계 저장
        param_stats = {
            'step': self.current_step,
            'total_parameters_mb': total_params / (1024 * 1024),
            'trainable_parameters_mb': trainable_params / (1024 * 1024),
            'non_trainable_parameters_mb': (total_params - trainable_params) / (1024 * 1024)
        }
        self.parameter_sizes.append(param_stats)
        
        return param_stats
    
    def log_memory_usage(self):
        """GPU 및 CPU 메모리 사용량을 로깅"""
        try:
            # GPU 메모리 사용량
            gpu_memory = 0
            if tf.config.list_physical_devices('GPU'):
                try:
                    # TensorFlow GPU 메모리 정보 가져오기
                    gpu_details = tf.config.experimental.get_memory_info('GPU:0')
                    gpu_memory = gpu_details['current'] / (1024 * 1024)  # MB 단위
                except:
                    # 대안 방법
                    try:
                        import subprocess
                        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
                                              capture_output=True, text=True)
                        if result.returncode == 0:
                            gpu_memory = float(result.stdout.strip())
                    except:
                        gpu_memory = 0
            
            # CPU 메모리 사용량
            cpu_memory = 0
            try:
                import psutil
                process = psutil.Process()
                cpu_memory = process.memory_info().rss / (1024 * 1024)  # MB 단위
            except:
                cpu_memory = 0
            
            # 메모리 사용량 저장
            memory_info = {
                'step': self.current_step,
                'gpu_memory_mb': gpu_memory,
                'cpu_memory_mb': cpu_memory
            }
            self.memory_usage.append(memory_info)
            
            # 파일에 로깅
            self.memory_log_file.write(f"{self.current_step},{gpu_memory:.2f},{cpu_memory:.2f}\n")
            self.memory_log_file.flush()
            
            return memory_info
            
        except Exception as e:
            print(f"메모리 사용량 로깅 오류: {e}")
            return {'step': self.current_step, 'gpu_memory_mb': 0, 'cpu_memory_mb': 0}
    
    def start_step(self, step):
        """새로운 스텝 시작"""
        self.current_step = step
        self.current_step_size = 0
        print(f"📊 Step {step} 텐서 프로파일링 시작")
    
    def end_step(self):
        """현재 스텝 종료 및 결과 저장"""
        step_size_mb = self.current_step_size / (1024 * 1024)
        self.step_tensor_sizes.append(step_size_mb)
        
        # 요약 정보 저장
        num_ops = len([detail for detail in self.tensor_details if detail['step'] == self.current_step])
        avg_tensor_size = step_size_mb / num_ops if num_ops > 0 else 0
        
        # 파일에 요약 정보 로깅
        self.summary_log_file.write(f"{self.current_step},{step_size_mb:.4f},{num_ops},{avg_tensor_size:.4f}\n")
        self.summary_log_file.flush()
        
        # Tiresias 스타일 텐서 사이즈 로깅
        self.tiresias_log_file.write(f"{self.current_step},{step_size_mb:.4f}\n")
        self.tiresias_log_file.flush()
        
        print(f"📊 Step {self.current_step} 완료 - TensorSize: {step_size_mb:.2f} MB")
        
        return step_size_mb
    
    def get_tiresias_tensorsize(self):
        """Tiresias 논문과 같은 방식으로 텐서 사이즈 계산"""
        if len(self.step_tensor_sizes) == 0:
            return 0
        
        # 처음 몇 스텝은 안정화 시간으로 제외 (워밍업)
        warmup_steps = min(3, len(self.step_tensor_sizes) // 4)
        stable_steps = self.step_tensor_sizes[warmup_steps:]
        
        if len(stable_steps) == 0:
            return np.mean(self.step_tensor_sizes) if self.step_tensor_sizes else 0
        
        # 안정화된 스텝들의 평균으로 tensorsize 계산
        tiresias_tensorsize = np.mean(stable_steps)
        
        return tiresias_tensorsize
    
    def calculate_tensor_skewness(self):
        """모델 텐서 크기들의 skewness(왜곡도) 계산"""
        try:
            # 모든 텐서 크기들 수집
            all_tensor_sizes = []
            
            for tensor_info in self.tensor_details:
                if tensor_info['size_bytes'] > 0:  # 0보다 큰 텐서만 포함
                    all_tensor_sizes.append(tensor_info['size_mb'])
            
            if len(all_tensor_sizes) < 3:  # skewness 계산을 위해 최소 3개 데이터 필요
                return 0.0
            
            # skewness 계산 (scipy.stats.skew 사용)
            tensor_skewness = stats.skew(all_tensor_sizes)
            
            return float(tensor_skewness)
            
        except Exception as e:
            print(f"Skewness 계산 오류: {e}")
            return 0.0
    
    def calculate_operation_skewness(self):
        """Operation별 텐서 크기들의 skewness 계산"""
        try:
            operation_skewness = {}
            
            for op_name, sizes in self.operation_tensor_sizes.items():
                if len(sizes) >= 3:  # 최소 3개 데이터 포인트 필요
                    sizes_mb = [size / (1024 * 1024) for size in sizes]
                    op_skewness = stats.skew(sizes_mb)
                    operation_skewness[op_name] = float(op_skewness)
            
            return operation_skewness
            
        except Exception as e:
            print(f"Operation skewness 계산 오류: {e}")
            return {}
    
    def calculate_layer_type_skewness(self):
        """레이어 타입별 텐서 크기들의 skewness 계산"""
        try:
            # 텐서 타입별로 그룹화
            type_sizes = {}
            
            for tensor_info in self.tensor_details:
                tensor_type = tensor_info['tensor_type']
                size_mb = tensor_info['size_mb']
                
                if tensor_type not in type_sizes:
                    type_sizes[tensor_type] = []
                
                if size_mb > 0:
                    type_sizes[tensor_type].append(size_mb)
            
            # 각 타입별 skewness 계산
            type_skewness = {}
            for tensor_type, sizes in type_sizes.items():
                if len(sizes) >= 3:
                    type_skewness[tensor_type] = float(stats.skew(sizes))
            
            return type_skewness
            
        except Exception as e:
            print(f"Layer type skewness 계산 오류: {e}")
            return {}
    
    def get_skewness_summary(self):
        """전체 skewness 분석 요약"""
        try:
            # 전체 모델 skewness
            model_skewness = self.calculate_tensor_skewness()
            
            # Operation별 skewness
            operation_skewness = self.calculate_operation_skewness()
            
            # 레이어 타입별 skewness
            layer_type_skewness = self.calculate_layer_type_skewness()
            
            # 통계 정보
            all_tensor_sizes = [info['size_mb'] for info in self.tensor_details if info['size_mb'] > 0]
            
            skewness_summary = {
                'model_skewness': model_skewness,
                'operation_skewness': operation_skewness,
                'layer_type_skewness': layer_type_skewness,
                'tensor_count': len(all_tensor_sizes),
                'mean_tensor_size_mb': np.mean(all_tensor_sizes) if all_tensor_sizes else 0,
                'std_tensor_size_mb': np.std(all_tensor_sizes) if all_tensor_sizes else 0,
                'min_tensor_size_mb': np.min(all_tensor_sizes) if all_tensor_sizes else 0,
                'max_tensor_size_mb': np.max(all_tensor_sizes) if all_tensor_sizes else 0
            }
            
            return skewness_summary
            
        except Exception as e:
            print(f"Skewness 요약 계산 오류: {e}")
            return {'model_skewness': 0.0}
    
    def log_skewness_analysis(self):
        """Skewness 분석 결과를 파일에 로깅"""
        try:
            skewness_summary = self.get_skewness_summary()
            
            # Skewness 분석 로그 파일
            skewness_log_file = os.path.join(self.log_dir, 'skewness_analysis.txt')
            with open(skewness_log_file, 'w') as f:
                f.write("=== Tensor Skewness Analysis ===\n")
                f.write(f"Model Skewness: {skewness_summary['model_skewness']:.2f}\n")
                f.write(f"Total Tensors: {skewness_summary['tensor_count']}\n")
                f.write(f"Mean Tensor Size: {skewness_summary['mean_tensor_size_mb']:.4f} MB\n")
                f.write(f"Std Tensor Size: {skewness_summary['std_tensor_size_mb']:.4f} MB\n")
                f.write(f"Min Tensor Size: {skewness_summary['min_tensor_size_mb']:.4f} MB\n")
                f.write(f"Max Tensor Size: {skewness_summary['max_tensor_size_mb']:.4f} MB\n")
                f.write("\n=== Layer Type Skewness ===\n")
                
                for layer_type, skewness in skewness_summary['layer_type_skewness'].items():
                    f.write(f"{layer_type}: {skewness:.2f}\n")
                
                f.write("\n=== Top 10 Operation Skewness ===\n")
                op_skewness = skewness_summary['operation_skewness']
                sorted_ops = sorted(op_skewness.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                
                for op_name, skewness in sorted_ops:
                    f.write(f"{op_name}: {skewness:.2f}\n")
            
            # JSON 형태로도 저장
            with open(os.path.join(self.log_dir, 'skewness_analysis.json'), 'w') as f:
                json.dump(skewness_summary, f, indent=2, default=str)
            
            return skewness_summary
            
        except Exception as e:
            print(f"Skewness 로깅 오류: {e}")
            return {'model_skewness': 0.0}
    
    def get_summary(self):
        """전체 프로파일링 결과 요약"""
        if not self.step_tensor_sizes:
            return {}
        
        tiresias_tensorsize = self.get_tiresias_tensorsize()
        
        # Skewness 분석 추가
        skewness_summary = self.get_skewness_summary()
        
        summary = {
            'total_steps': len(self.step_tensor_sizes),
            'tiresias_tensorsize_mb': tiresias_tensorsize,
            'avg_step_tensorsize_mb': np.mean(self.step_tensor_sizes),
            'max_step_tensorsize_mb': np.max(self.step_tensor_sizes),
            'min_step_tensorsize_mb': np.min(self.step_tensor_sizes),
            'std_step_tensorsize_mb': np.std(self.step_tensor_sizes),
            'total_operations': len(self.tensor_details),
            'step_tensor_sizes': self.step_tensor_sizes,
            'model_skewness': skewness_summary['model_skewness'],  # 추가
            'skewness_analysis': skewness_summary  # 추가
        }
        
        # Operation별 통계
        op_stats = {}
        for op_name, sizes in self.operation_tensor_sizes.items():
            op_stats[op_name] = {
                'total_size_mb': sum(sizes) / (1024 * 1024),
                'avg_size_mb': np.mean(sizes) / (1024 * 1024),
                'count': len(sizes)
            }
        
        summary['operation_stats'] = op_stats
        
        return summary
    
    def save_final_results(self):
        """최종 결과를 파일에 저장"""
        summary = self.get_summary()
        
        # JSON 형태로 저장
        with open(os.path.join(self.log_dir, 'final_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Tiresias 결과 저장
        tiresias_result = {
            'model': 'whisper_small',
            'tensorsize_mb': summary['tiresias_tensorsize_mb'],
            'skewness': summary['model_skewness'],  # 추가
            'total_steps': summary['total_steps'],
            'measurement_method': 'Tiresias_style'
        }
        
        with open(os.path.join(self.log_dir, 'tiresias_result.json'), 'w') as f:
            json.dump(tiresias_result, f, indent=2)
        
        # 레거시 포맷으로 skewness 결과 저장
        with open(os.path.join(self.log_dir, 'legacy_skewness_result.txt'), 'w') as f:
            f.write("model,skewness\n")
            f.write(f"whisper_small,{summary['model_skewness']:.1f}\n")
        
        # Skewness 분석 로깅
        self.log_skewness_analysis()
        
        return summary
    
    def _calculate_tensor_size(self, tensor):
        """텐서의 메모리 사이즈를 바이트 단위로 계산"""
        try:
            if tensor is None:
                return 0
            
            # 텐서의 모든 요소 개수 계산
            if hasattr(tensor, 'shape'):
                total_elements = tf.size(tensor).numpy() if hasattr(tf.size(tensor), 'numpy') else 1
                for dim in tensor.shape:
                    if dim is not None:
                        total_elements = total_elements if hasattr(tf.size(tensor), 'numpy') else total_elements * int(dim)
            else:
                total_elements = 1
            
            # 데이터 타입별 바이트 크기
            dtype_size = tensor.dtype.size if hasattr(tensor, 'dtype') else 4  # 기본값 4바이트
            
            return int(total_elements * dtype_size)
        except Exception as e:
            print(f"텐서 크기 계산 오류: {e}")
            return 0
    
    def close(self):
        """프로파일러 종료 및 파일 닫기"""
        try:
            self.tensor_log_file.close()
            self.memory_log_file.close()
            self.summary_log_file.close()
            self.tiresias_log_file.close()
            print(f"🔍 TensorProfiler 종료됨")
        except:
            pass


class TensorLoggingMixin:
    """레이어에 텐서 로깅 기능을 추가하는 믹스인 클래스"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profiler = None
    
    def set_profiler(self, profiler):
        """프로파일러 설정"""
        self.profiler = profiler
    
    def log_tensor_if_profiler(self, tensor, name, tensor_type="activation"):
        """프로파일러가 설정된 경우 텐서 로깅"""
        if self.profiler is not None:
            return self.profiler.log_tensor_size(tensor, name, tensor_type)
        return 0


class TensorSizeMonitor:
    """텐서 사이즈를 모니터링하는 클래스"""
    
    def __init__(self):
        self.current_step_size = 0
        self.total_size = 0
        self.op_sizes = {}
    
    def calculate_tensor_size(self, tensor):
        """텐서의 메모리 사이즈를 바이트 단위로 계산"""
        if tensor is None:
            return 0
        
        try:
            # 텐서의 shape과 dtype을 이용해 메모리 사이즈 계산
            shape = tf.shape(tensor).numpy() if hasattr(tf.shape(tensor), 'numpy') else tensor.shape
            dtype_size = tensor.dtype.size
            
            total_elements = 1
            for dim in shape:
                total_elements *= int(dim)
            
            return total_elements * dtype_size
        except:
            # shape을 가져올 수 없는 경우 추정
            try:
                return tf.size(tensor).numpy() * tensor.dtype.size
            except:
                return 0
    
    def track_tensor(self, tensor, operation_name="unknown"):
        """텐서를 추적하고 사이즈를 기록"""
        size = self.calculate_tensor_size(tensor)
        
        if operation_name not in self.op_sizes:
            self.op_sizes[operation_name] = []
        
        self.op_sizes[operation_name].append(size)
        self.current_step_size += size
        
        return size
    
    def reset_step(self):
        """스텝 초기화"""
        self.current_step_size = 0
    
    def get_step_summary(self):
        """현재 스텝의 텐서 사이즈 요약"""
        return {
            'step_total_size': self.current_step_size,
            'operation_sizes': {op: sum(sizes) for op, sizes in self.op_sizes.items()}
        }


# 전역 텐서 모니터
tensor_monitor = TensorSizeMonitor()


class WhisperConfig:
    def __init__(self):
        # 모델 크기 설정 (Whisper-small 기준)
        self.d_model = 768  # 모델 차원
        self.encoder_layers = 4  # 인코더 레이어 수
        self.encoder_attention_heads = 12  # 인코더 어텐션 헤드 수
        self.decoder_layers = 4  # 디코더 레이어 수
        self.decoder_attention_heads = 12  # 디코더 어텐션 헤드 수
        self.d_ff = 3072  # 피드포워드 네트워크 차원
        
        # 인코더 설정
        self.n_mels = 80  # 멜 스펙트로그램 특징 수
        self.n_ctx = 1500  # 최대 컨텍스트 길이 (30초 오디오)
        
        # 디코더 설정
        self.vocab_size = 51865  # 전체 어휘 크기
        self.max_target_positions = 448  # 최대 출력 시퀀스 길이
        
        # 학습 관련 설정
        self.dropout = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.0
        self.activation_function = "gelu"
        
        # 기타 설정
        self.layer_norm_eps = 1e-5
        self.init_std = 0.02
        
        # 특수 토큰 설정
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        
        # 추가 설정
        self.use_cache = True
        self.decoder_start_token_id = 50257  # <|startoftranscript|>


# 위치 인코딩
class PositionalEncoding(tf.keras.layers.Layer, TensorLoggingMixin):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        
        # 위치 인코딩 행렬 계산
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        # 짝수 인덱스에는 sin, 홀수 인덱스에는 cos
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.convert_to_tensor(pe, dtype=tf.float32)
        self.pe = tf.expand_dims(self.pe, 0)  # [1, max_len, d_model]
    
    def call(self, x):
        # x: [batch_size, seq_len, d_model]
        self.log_tensor_if_profiler(x, "positional_encoding_input")
        tensor_monitor.track_tensor(x, "positional_encoding_input")
        seq_len = tf.shape(x)[1]
        result = x + self.pe[:, :seq_len, :]
        self.log_tensor_if_profiler(result, "positional_encoding_output")
        tensor_monitor.track_tensor(result, "positional_encoding_output")
        return result


# 멀티헤드 어텐션 구현
class MultiHeadAttention(tf.keras.layers.Layer, TensorLoggingMixin):
    def __init__(self, config, is_decoder=False, is_cross_attention=False):
        super(MultiHeadAttention, self).__init__()
        self.is_decoder = is_decoder
        self.is_cross_attention = is_cross_attention
        
        if is_decoder:
            self.num_heads = config.decoder_attention_heads
            self.d_model = config.d_model
        else:
            self.num_heads = config.encoder_attention_heads
            self.d_model = config.d_model
            
        self.head_dim = self.d_model // self.num_heads
        self.scaling = self.head_dim ** -0.5
        
        self.k_proj = tf.keras.layers.Dense(self.d_model, use_bias=True)
        self.v_proj = tf.keras.layers.Dense(self.d_model, use_bias=True)
        self.q_proj = tf.keras.layers.Dense(self.d_model, use_bias=True)
        self.out_proj = tf.keras.layers.Dense(self.d_model, use_bias=True)
        
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)
    
    def _reshape(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, num_heads, head_dim]
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.head_dim))
        
        # [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, hidden_states, key_value_states=None, attention_mask=None, 
             past_key_value=None, layer_head_mask=None, training=False):
        """
        hidden_states: 쿼리 텐서 [batch_size, seq_len, d_model]
        key_value_states: 키/값 텐서 (cross-attention 사용 시) [batch_size, kv_seq_len, d_model]
        attention_mask: 어텐션 마스크 [batch_size, seq_len, kv_seq_len]
        past_key_value: 캐싱된 키/값 (디코더에서 사용)
        """
        self.log_tensor_if_profiler(hidden_states, "attention_hidden_states_input")
        tensor_monitor.track_tensor(hidden_states, "attention_hidden_states_input")
        
        batch_size = tf.shape(hidden_states)[0]
        seq_len = tf.shape(hidden_states)[1]
        
        # cross-attention인 경우 key, value는 인코더 출력, query는 디코더 상태
        is_cross_attention = key_value_states is not None
        
        if is_cross_attention:
            # cross-attention인 경우 key_value_states에서 key와 value 추출
            key_states = self._reshape(self.k_proj(key_value_states))  # [batch, num_heads, kv_seq_len, head_dim]
            value_states = self._reshape(self.v_proj(key_value_states))  # [batch, num_heads, kv_seq_len, head_dim]
            self.log_tensor_if_profiler(key_states, "cross_attention_key_states")
            self.log_tensor_if_profiler(value_states, "cross_attention_value_states")
            tensor_monitor.track_tensor(key_states, "cross_attention_key_states")
            tensor_monitor.track_tensor(value_states, "cross_attention_value_states")
            kv_seq_len = tf.shape(key_states)[2]
        elif past_key_value is not None:
            # 과거 키/값이 있는 경우 (디코더의 auto-regressive 생성 시)
            key_states = self._reshape(self.k_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
            value_states = self._reshape(self.v_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
            
            # 과거 키/값과 현재 키/값 연결
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
            self.log_tensor_if_profiler(key_states, "past_key_states")
            self.log_tensor_if_profiler(value_states, "past_value_states")
            tensor_monitor.track_tensor(key_states, "past_key_states")
            tensor_monitor.track_tensor(value_states, "past_value_states")
            kv_seq_len = tf.shape(key_states)[2]
        else:
            # 일반적인 self-attention
            key_states = self._reshape(self.k_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
            value_states = self._reshape(self.v_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
            self.log_tensor_if_profiler(key_states, "self_attention_key_states")
            self.log_tensor_if_profiler(value_states, "self_attention_value_states")
            tensor_monitor.track_tensor(key_states, "self_attention_key_states")
            tensor_monitor.track_tensor(value_states, "self_attention_value_states")
            kv_seq_len = seq_len
        
        # 항상 쿼리는 현재 hidden_states에서 계산
        query_states = self._reshape(self.q_proj(hidden_states) * self.scaling)  # [batch, num_heads, seq_len, head_dim]
        self.log_tensor_if_profiler(query_states, "attention_query_states")
        tensor_monitor.track_tensor(query_states, "attention_query_states")
        
        # 현재 키/값 저장 (디코더에서 캐싱 시 사용)
        past_key_value = (key_states, value_states) if self.is_decoder else None
        
        # 어텐션 스코어 계산: [batch, num_heads, seq_len, kv_seq_len]
        attention_scores = tf.matmul(query_states, key_states, transpose_b=True)
        self.log_tensor_if_profiler(attention_scores, "attention_scores")
        tensor_monitor.track_tensor(attention_scores, "attention_scores")
        
        # 어텐션 마스크 적용 (존재하는 경우)
        if attention_mask is not None:
            # 마스크 확장 및 적용 (마스크가 0인 위치는 -inf로 설정)
            attention_mask = tf.cast(attention_mask, tf.float32)
            attention_mask = (1.0 - attention_mask) * -1e9
            attention_scores = attention_scores + attention_mask
            self.log_tensor_if_profiler(attention_mask, "attention_mask")
            tensor_monitor.track_tensor(attention_mask, "attention_mask")
        
        # 소프트맥스 적용
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        self.log_tensor_if_profiler(attention_probs, "attention_probs")
        tensor_monitor.track_tensor(attention_probs, "attention_probs")
        
        # 드롭아웃 적용
        attention_probs = self.dropout(attention_probs, training=training)
        
        # 헤드 마스크 적용 (필요한 경우)
        if layer_head_mask is not None:
            attention_probs = attention_probs * tf.expand_dims(tf.expand_dims(layer_head_mask, -1), -1)
        
        # 어텐션 출력 계산
        attention_output = tf.matmul(attention_probs, value_states)  # [batch, num_heads, seq_len, head_dim]
        self.log_tensor_if_profiler(attention_output, "attention_output_raw")
        tensor_monitor.track_tensor(attention_output, "attention_output_raw")
        
        # 출력 형태 변환
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])  # [batch, seq_len, num_heads, head_dim]
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.d_model))  # [batch, seq_len, d_model]
        
        # 최종 선형 변환
        attention_output = self.out_proj(attention_output)
        self.log_tensor_if_profiler(attention_output, "attention_output_final")
        tensor_monitor.track_tensor(attention_output, "attention_output_final")
        
        return attention_output, attention_probs, past_key_value


# 피드포워드 네트워크
class FeedForward(tf.keras.layers.Layer, TensorLoggingMixin):
    def __init__(self, config, is_decoder=False):
        super(FeedForward, self).__init__()
        if is_decoder:
            d_model = config.d_model
            d_ff = config.d_ff
            dropout = config.dropout
            activation_dropout = config.activation_dropout
        else:
            d_model = config.d_model
            d_ff = config.d_ff
            dropout = config.dropout
            activation_dropout = config.activation_dropout
        
        self.fc1 = tf.keras.layers.Dense(d_ff, use_bias=True)
        self.activation_fn = tf.keras.activations.gelu
        self.activation_dropout = tf.keras.layers.Dropout(activation_dropout)
        self.fc2 = tf.keras.layers.Dense(d_model, use_bias=True)
        self.dropout = tf.keras.layers.Dropout(dropout)
    
    def call(self, hidden_states, training=False):
        self.log_tensor_if_profiler(hidden_states, "feedforward_input")
        tensor_monitor.track_tensor(hidden_states, "feedforward_input")
        
        hidden_states = self.fc1(hidden_states)
        self.log_tensor_if_profiler(hidden_states, "feedforward_fc1_output")
        tensor_monitor.track_tensor(hidden_states, "feedforward_fc1_output")
        
        hidden_states = self.activation_fn(hidden_states)
        self.log_tensor_if_profiler(hidden_states, "feedforward_activation_output")
        tensor_monitor.track_tensor(hidden_states, "feedforward_activation_output")
        
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        self.log_tensor_if_profiler(hidden_states, "feedforward_fc2_output")
        tensor_monitor.track_tensor(hidden_states, "feedforward_fc2_output")
        
        hidden_states = self.dropout(hidden_states, training=training)
        self.log_tensor_if_profiler(hidden_states, "feedforward_final_output")
        tensor_monitor.track_tensor(hidden_states, "feedforward_final_output")
        
        return hidden_states


# 인코더 레이어
class WhisperEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(WhisperEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(config, is_decoder=False)
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.feed_forward = FeedForward(config, is_decoder=False)
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
    
    def call(self, hidden_states, attention_mask=None, layer_head_mask=None, training=False):
        tensor_monitor.track_tensor(hidden_states, "encoder_layer_input")
        
        # Self Attention
        residual = hidden_states
        tensor_monitor.track_tensor(residual, "encoder_residual_1")
        
        hidden_states = self.self_attn_layer_norm(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "encoder_layer_norm_1")
        
        attention_output, attention_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            training=training
        )
        hidden_states = residual + attention_output
        tensor_monitor.track_tensor(hidden_states, "encoder_after_attention")
        
        # Feed Forward
        residual = hidden_states
        tensor_monitor.track_tensor(residual, "encoder_residual_2")
        
        hidden_states = self.final_layer_norm(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "encoder_layer_norm_2")
        
        feed_forward_output = self.feed_forward(hidden_states, training=training)
        hidden_states = residual + feed_forward_output
        tensor_monitor.track_tensor(hidden_states, "encoder_layer_output")
        
        return hidden_states, attention_weights


# 디코더 레이어
class WhisperDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(WhisperDecoderLayer, self).__init__()
        # Self Attention
        self.self_attn = MultiHeadAttention(config, is_decoder=True)
        self.self_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        
        # Cross Attention
        self.encoder_attn = MultiHeadAttention(config, is_decoder=True, is_cross_attention=True)
        self.encoder_attn_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        
        # Feed Forward
        self.feed_forward = FeedForward(config, is_decoder=True)
        self.final_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
    
    def call(self, hidden_states, encoder_hidden_states, 
             attention_mask=None, encoder_attention_mask=None,
             layer_head_mask=None, cross_attn_layer_head_mask=None,
             past_key_value=None, training=False):
        
        tensor_monitor.track_tensor(hidden_states, "decoder_layer_input")
        tensor_monitor.track_tensor(encoder_hidden_states, "decoder_encoder_hidden_states")
        
        # 캐시된 과거 키/값 분리
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        cross_attn_past_key_value = past_key_value[2:] if past_key_value is not None else None
        
        # Self Attention
        residual = hidden_states
        tensor_monitor.track_tensor(residual, "decoder_residual_1")
        
        hidden_states = self.self_attn_layer_norm(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "decoder_layer_norm_1")
        
        attention_output, self_attention_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            training=training
        )
        
        hidden_states = residual + attention_output
        tensor_monitor.track_tensor(hidden_states, "decoder_after_self_attention")
        
        # Cross Attention
        residual = hidden_states
        tensor_monitor.track_tensor(residual, "decoder_residual_2")
        
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "decoder_layer_norm_2")
        
        cross_attention_output, cross_attention_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            training=training
        )
        
        hidden_states = residual + cross_attention_output
        tensor_monitor.track_tensor(hidden_states, "decoder_after_cross_attention")
        
        # Feed Forward
        residual = hidden_states
        tensor_monitor.track_tensor(residual, "decoder_residual_3")
        
        hidden_states = self.final_layer_norm(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "decoder_layer_norm_3")
        
        feed_forward_output = self.feed_forward(hidden_states, training=training)
        hidden_states = residual + feed_forward_output
        tensor_monitor.track_tensor(hidden_states, "decoder_layer_output")
        
        # 현재 레이어의 키/값 모음
        present_key_value = present_key_value + cross_attn_present_key_value if present_key_value is not None else None
        
        return hidden_states, self_attention_weights, cross_attention_weights, present_key_value


# 오디오 인코더
class WhisperEncoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(WhisperEncoder, self).__init__()
        self.config = config
        
        # 오디오 특징 추출 및 임베딩
        self.conv1 = tf.keras.layers.Conv1D(config.d_model, kernel_size=3, strides=1, padding="same")
        self.conv2 = tf.keras.layers.Conv1D(config.d_model, kernel_size=3, strides=2, padding="same")
        self.positional_embedding = PositionalEncoding(config.d_model, config.n_ctx)
        
        # 드롭아웃
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        
        # 인코더 레이어
        self.layers = [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        
        # 레이어 정규화
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
    
    def call(self, input_features, attention_mask=None, layer_head_mask=None, training=False):
        """
        input_features: 멜 스펙트로그램 특징 [batch_size, n_mels, seq_len]
        """
        tensor_monitor.track_tensor(input_features, "encoder_input_features")
        
        # 차원 변환 (채널 마지막)
        input_features = tf.transpose(input_features, perm=[0, 2, 1])  # [batch, seq_len, n_mels]
        tensor_monitor.track_tensor(input_features, "encoder_transposed_features")
        
        # 컨볼루션 레이어 적용
        hidden_states = self.conv1(input_features)
        tensor_monitor.track_tensor(hidden_states, "encoder_conv1_output")
        
        hidden_states = tf.keras.activations.gelu(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "encoder_conv1_gelu")
        
        hidden_states = self.conv2(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "encoder_conv2_output")
        
        hidden_states = tf.keras.activations.gelu(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "encoder_conv2_gelu")
        
        # 위치 인코딩 추가
        hidden_states = self.positional_embedding(hidden_states)
        
        # 드롭아웃
        hidden_states = self.dropout(hidden_states, training=training)
        tensor_monitor.track_tensor(hidden_states, "encoder_after_dropout")
        
        # 레이어별 처리
        all_hidden_states = ()
        all_self_attentions = ()
        
        for i, layer in enumerate(self.layers):
            all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 레이어 헤드 마스크 가져오기
            layer_head_mask_i = layer_head_mask[i] if layer_head_mask is not None else None
            
            # 레이어 호출
            hidden_states, self_attn_weights = layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=layer_head_mask_i,
                training=training
            )
            
            all_self_attentions = all_self_attentions + (self_attn_weights,)
            tensor_monitor.track_tensor(hidden_states, f"encoder_layer_{i}_output")
        
        # 최종 레이어 정규화
        hidden_states = self.layer_norm(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "encoder_final_output")
        
        # 최종 결과 반환
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions
        }


# 텍스트 디코더
class WhisperDecoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(WhisperDecoder, self).__init__()
        self.config = config
        
        # 토큰 임베딩
        self.embed_tokens = tf.keras.layers.Embedding(config.vocab_size, config.d_model)
        self.positional_embedding = PositionalEncoding(config.d_model, config.max_target_positions)
        
        # 드롭아웃
        self.dropout = tf.keras.layers.Dropout(config.dropout)
        
        # 디코더 레이어
        self.layers = [WhisperDecoderLayer(config) for _ in range(config.decoder_layers)]
        
        # 레이어 정규화
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
    
    def call(self, input_ids, encoder_hidden_states,
             attention_mask=None, encoder_attention_mask=None,
             layer_head_mask=None, cross_attn_layer_head_mask=None,
             past_key_values=None, use_cache=False, training=False):
        """
        input_ids: 입력 토큰 ID [batch_size, seq_len]
        encoder_hidden_states: 인코더 출력 [batch_size, enc_seq_len, d_model]
        """
        tensor_monitor.track_tensor(input_ids, "decoder_input_ids")
        
        batch_size, seq_length = tf.shape(input_ids)[0], tf.shape(input_ids)[1]
        
        # 입력 토큰 임베딩
        inputs_embeds = self.embed_tokens(input_ids)
        tensor_monitor.track_tensor(inputs_embeds, "decoder_token_embeddings")
        
        # 위치 인코딩 추가
        hidden_states = self.positional_embedding(inputs_embeds)
        
        # 드롭아웃
        hidden_states = self.dropout(hidden_states, training=training)
        tensor_monitor.track_tensor(hidden_states, "decoder_after_dropout")
        
        # 어텐션 마스크 확인 및 생성
        if attention_mask is None:
            # 인과적 마스크 생성 (자기 자신과 이전 위치만 볼 수 있음)
            attention_mask = 1.0 - tf.linalg.band_part(
                tf.ones((seq_length, seq_length)), -1, 0)
            attention_mask = tf.expand_dims(attention_mask, 0)  # [1, seq_len, seq_len]
            tensor_monitor.track_tensor(attention_mask, "decoder_causal_mask")
        
        # 초기화
        all_hidden_states = ()
        all_self_attentions = ()
        all_cross_attentions = ()
        present_key_values = () if use_cache else None
        
        # 레이어별 처리
        for i, layer in enumerate(self.layers):
            all_hidden_states = all_hidden_states + (hidden_states,)
            
            # 과거 키/값 가져오기
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # 마스크 가져오기
            layer_head_mask_i = layer_head_mask[i] if layer_head_mask is not None else None
            cross_attn_layer_head_mask_i = cross_attn_layer_head_mask[i] if cross_attn_layer_head_mask is not None else None
            
            # 레이어 호출
            hidden_states, self_attn_weights, cross_attn_weights, present_key_value = layer(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask_i,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask_i,
                past_key_value=past_key_value,
                training=training
            )
            
            # 결과 저장
            all_self_attentions = all_self_attentions + (self_attn_weights,)
            all_cross_attentions = all_cross_attentions + (cross_attn_weights,)
            
            tensor_monitor.track_tensor(hidden_states, f"decoder_layer_{i}_output")
            
            if use_cache:
                present_key_values = present_key_values + (present_key_value,)
        
        # 최종 레이어 정규화
        hidden_states = self.layer_norm(hidden_states)
        tensor_monitor.track_tensor(hidden_states, "decoder_final_output")
        
        # 결과 반환
        return {
            "last_hidden_state": hidden_states,
            "past_key_values": present_key_values,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
            "cross_attentions": all_cross_attentions
        }


# 전체 Whisper 모델
class WhisperModel(tf.keras.Model):
    def __init__(self, config):
        super(WhisperModel, self).__init__()
        self.config = config
        
        # 인코더와 디코더 초기화
        self.encoder = WhisperEncoder(config)
        self.decoder = WhisperDecoder(config)
    
    def call(self, input_features, decoder_input_ids=None,
             attention_mask=None, decoder_attention_mask=None,
             encoder_outputs=None, past_key_values=None,
             use_cache=None, return_dict=True, training=False):
        """
        input_features: 멜 스펙트로그램 특징 [batch_size, n_mels, seq_len]
        decoder_input_ids: 디코더 입력 토큰 ID [batch_size, seq_len]
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        # 인코더 처리 (인코더 출력이 제공되지 않은 경우)
        if encoder_outputs is None and input_features is not None:
            encoder_outputs = self.encoder(
                input_features,
                attention_mask=attention_mask,
                training=training
            )
        
        # 인코더 출력
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        
        # 디코더 입력이 제공되지 않은 경우 시작 토큰 생성
        if decoder_input_ids is None:
            batch_size = tf.shape(encoder_hidden_states)[0]
            decoder_input_ids = tf.fill((batch_size, 1), self.config.decoder_start_token_id)
        
        # 디코더 호출
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            training=training
        )
        
        if not return_dict:
            return (
                decoder_outputs["last_hidden_state"],
                decoder_outputs["past_key_values"],
                encoder_outputs["last_hidden_state"]
            )
        
        return {
            "last_hidden_state": decoder_outputs["last_hidden_state"],
            "past_key_values": decoder_outputs["past_key_values"],
            "encoder_last_hidden_state": encoder_outputs["last_hidden_state"],
            "encoder_hidden_states": encoder_outputs.get("hidden_states", None),
            "encoder_attentions": encoder_outputs.get("attentions", None),
            "decoder_hidden_states": decoder_outputs.get("hidden_states", None),
            "decoder_attentions": decoder_outputs.get("attentions", None),
            "cross_attentions": decoder_outputs.get("cross_attentions", None)
        }


# 언어 모델링을 위한 Whisper 모델
class WhisperForConditionalGeneration(tf.keras.Model):
    def __init__(self, config):
        super(WhisperForConditionalGeneration, self).__init__()
        self.config = config
        
        # 기본 Whisper 모델
        self.model = WhisperModel(config)
        
        # 언어 모델링 헤드
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, use_bias=False)
    
    def call(self, input_features, decoder_input_ids=None,
             attention_mask=None, decoder_attention_mask=None,
             encoder_outputs=None, past_key_values=None,
             labels=None, use_cache=None, return_dict=True, training=False):
        """
        input_features: 멜 스펙트로그램 특징 [batch_size, n_mels, seq_len]
        decoder_input_ids: 디코더 입력 토큰 ID [batch_size, seq_len]
        labels: 타겟 토큰 ID [batch_size, seq_len]
        """
        # 디코더 입력이 제공되지 않은 경우 레이블에서 생성
        if decoder_input_ids is None and labels is not None:
            # 레이블의 오른쪽으로 시프트하여 디코더 입력 생성 (teacher forcing)
            decoder_input_ids = tf.pad(
                labels[:, :-1], 
                [[0, 0], [1, 0]], 
                constant_values=self.config.decoder_start_token_id
            )
            tensor_monitor.track_tensor(decoder_input_ids, "shifted_decoder_input_ids")
        
        if labels is not None:
            tensor_monitor.track_tensor(labels, "training_labels")
        
        # Whisper 모델 호출
        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
            training=training
        )
        
        # 언어 모델링 헤드 적용
        lm_logits = self.lm_head(outputs["last_hidden_state"])
        tensor_monitor.track_tensor(lm_logits, "lm_head_logits")
        
        # 손실 계산 (학습 중이고 레이블이 제공된 경우)
        loss = None
        if training and labels is not None:
            # 손실 계산을 위해 레이블 시프트 (teacher forcing)
            shift_labels = labels[:, 1:]
            shift_logits = lm_logits[:, :-1, :]
            
            tensor_monitor.track_tensor(shift_labels, "shift_labels")
            tensor_monitor.track_tensor(shift_logits, "shift_logits")
            
            # 손실 계산 (교차 엔트로피)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )
            
            loss = loss_fn(shift_labels, shift_logits)
            tensor_monitor.track_tensor(loss, "raw_loss")
            
            # 패딩 토큰 마스킹 (패딩 토큰은 손실 계산에서 제외)
            if decoder_attention_mask is not None:
                loss = loss * tf.cast(decoder_attention_mask[:, :-1], dtype=loss.dtype)
                loss = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(decoder_attention_mask[:, :-1], dtype=loss.dtype))
            else:
                loss = tf.reduce_mean(loss)
            
            tensor_monitor.track_tensor(loss, "final_loss")
        
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": lm_logits,
            "past_key_values": outputs["past_key_values"],
            "encoder_last_hidden_state": outputs["encoder_last_hidden_state"],
            "encoder_hidden_states": outputs.get("encoder_hidden_states", None),
            "encoder_attentions": outputs.get("encoder_attentions", None),
            "decoder_hidden_states": outputs.get("decoder_hidden_states", None),
            "decoder_attentions": outputs.get("decoder_attentions", None),
            "cross_attentions": outputs.get("cross_attentions", None)
        }
    
    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, 
                                     attention_mask=None, use_cache=None, encoder_outputs=None,
                                     **kwargs):
        """
        생성(디코딩) 중에 입력을 준비하는 도우미 함수
        """
        # 과거 키/값이 제공되면 마지막 토큰만 입력으로 사용
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
        
        return {
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache
        }
    
    def generate(self, input_features, max_length=None, min_length=None, 
                 num_beams=None, temperature=1.0, top_k=None, top_p=None,
                 repetition_penalty=None, attention_mask=None, **kwargs):
        """
        텍스트 생성 함수
        """
        max_length = max_length if max_length is not None else self.config.max_target_positions
        min_length = min_length if min_length is not None else 0
        num_beams = num_beams if num_beams is not None else 1
        temperature = temperature if temperature is not None else 1.0
        top_k = top_k if top_k is not None else 50
        top_p = top_p if top_p is not None else 1.0
        repetition_penalty = repetition_penalty if repetition_penalty is not None else 1.0
        
        batch_size = tf.shape(input_features)[0]
        
        # 인코더 출력 계산
        encoder_outputs = self.model.encoder(
            input_features=input_features,
            attention_mask=attention_mask,
            training=False
        )
        
        # 디코더 시작 토큰 설정
        decoder_input_ids = tf.fill((batch_size, 1), self.config.decoder_start_token_id)
        
        # 생성 루프 (간단한 그리디 디코딩 구현)
        for _ in range(max_length):
            # 모델 호출
            outputs = self.model(
                input_features=None,  # 인코더는 이미 실행됨
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=True,
                training=False
            )
            
            # 다음 토큰 예측
            next_token_logits = outputs["logits"][:, -1, :]
            
            # 온도 적용
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Top-k 샘플링
            if top_k > 0:
                indices_to_remove = tf.argsort(next_token_logits, direction='DESCENDING')[:, top_k:]
                indices_to_remove = tf.expand_dims(indices_to_remove, -1)
                next_token_logits = tf.tensor_scatter_nd_update(
                    next_token_logits,
                    indices_to_remove,
                    tf.fill([tf.shape(indices_to_remove)[0], tf.shape(indices_to_remove)[1]], -float('inf'))
                )
            
            # 다음 토큰 선택
            if num_beams > 1:
                # 빔 서치 구현은 복잡하므로 여기서는 생략
                pass
            else:
                # 그리디 디코딩
                next_tokens = tf.argmax(next_token_logits, axis=-1, output_type=tf.int32)
            
            # EOS 토큰 체크
            eos_tokens = tf.equal(next_tokens, self.config.eos_token_id)
            
            # 새 토큰 추가
            decoder_input_ids = tf.concat([decoder_input_ids, tf.expand_dims(next_tokens, -1)], axis=-1)
            
            # 모든 시퀀스가 EOS 토큰에 도달했는지 확인
            if tf.reduce_all(eos_tokens):
                break
        
        return decoder_input_ids
    
    def train_step(self, data):
        """
        학습 스텝 구현
        """
        features, labels = data
        
        with tf.GradientTape() as tape:
            # 모델 순전파
            outputs = self(features, labels=labels, training=True)
            loss = outputs["loss"]
        
        # 그래디언트 계산 및 적용
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 메트릭 업데이트
        self.compiled_metrics.update_state(labels, outputs["logits"])
        
        # 결과 반환
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        
        return results


# ============ 유틸리티 함수 및 학습 관련 코드 ============ #

# 오디오 특징 추출 함수
def extract_fbank_features(waveform, sample_rate=16000, n_mels=80, n_fft=400, hop_length=160):
    """
    오디오 파형에서 멜 스펙트로그램 특징 추출
    """
    # TensorFlow로 FFT 계산
    stfts = tf.signal.stft(
        waveform, 
        frame_length=n_fft, 
        frame_step=hop_length, 
        fft_length=n_fft
    )
    
    # 스펙트럼의 파워 계산
    power_spectrograms = tf.math.square(tf.abs(stfts))
    
    # 멜 필터뱅크 생성
    num_spectrogram_bins = n_fft // 2 + 1
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        n_mels, num_spectrogram_bins, sample_rate, 0, sample_rate // 2
    )
    
    # 멜 스펙트로그램 계산
    mel_spectrograms = tf.tensordot(power_spectrograms, linear_to_mel_weight_matrix, 1)
    
    # 로그 멜 스펙트로그램 계산
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)
    
    return log_mel_spectrograms


# 텍스트 전처리 함수
def preprocess_text(text, tokenizer):
    """
    텍스트를 토큰화하고 모델 입력 형식으로 변환
    """
    # 텍스트 정규화 및 토큰화
    tokens = tokenizer.encode(text)
    
    # 시작 및 종료 토큰 추가
    tokens = [tokenizer.bos_token_id] + tokens + [tokenizer.eos_token_id]
    
    return tokens


# 더미 데이터셋 생성
def create_dummy_dataset(batch_size, n_mels=80, seq_len=3000, max_target_length=100):
    """
    학습용 더미 오디오-텍스트 데이터셋을 생성합니다.
    """
    # 충분히 큰 더미 데이터셋
    num_samples = 50  # 데이터셋 크기
    
    # 더미 특성 (멜 스펙트로그램) 생성
    dummy_features = np.random.randn(num_samples, n_mels, seq_len).astype(np.float32)
    
    # 더미 레이블 (토큰 ID) 생성 - 첫 번째 토큰은 BOS(1), 마지막 토큰은 EOS(2)로 설정
    dummy_labels = np.zeros((num_samples, max_target_length), dtype=np.int32)
    
    # 각 시퀀스의 실제 길이 (50~90 사이의 무작위 길이)
    sequence_lengths = np.random.randint(50, 90, size=num_samples)
    
    for i in range(num_samples):
        # 첫 번째 토큰은 BOS 토큰
        dummy_labels[i, 0] = 1  # BOS token
        
        # 중간 토큰은 랜덤 (3부터 100 사이의 값)
        length = sequence_lengths[i]
        dummy_labels[i, 1:length-1] = np.random.randint(3, 100, size=length-2)
        
        # 마지막 토큰은 EOS 토큰
        dummy_labels[i, length-1] = 2  # EOS token
    
    # TensorFlow 데이터셋 생성
    dataset = tf.data.Dataset.from_tensor_slices((dummy_features, dummy_labels))
    
    # 배치 설정 및 반복
    return dataset.batch(batch_size).repeat()


@tf.function
def distributed_train_step(strategy, model, dist_inputs, optimizer):
    """분산 학습을 위한 스텝 함수 (텐서 사이즈 측정 포함)"""
    
    def train_step(inputs):
        features, labels = inputs
        
        # 스텝 시작 시 텐서 모니터 리셋
        tensor_monitor.reset_step()
        
        with tf.GradientTape() as tape:
            # 모델 호출
            try:
                outputs = model(features, labels=labels, training=True)
                loss = outputs["loss"]
                
                # 그래디언트 계산 및 적용
                gradients = tape.gradient(loss, model.trainable_variables)
                
                # 그래디언트 텐서 사이즈 측정
                for i, grad in enumerate(gradients):
                    if grad is not None:
                        tensor_monitor.track_tensor(grad, f"gradient_{i}")
                
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                return loss
            except Exception as e:
                print(f"학습 스텝 실행 중 오류 발생: {e}")
                # 형태 디버깅 출력
                print(f"features.shape: {features.shape}")
                print(f"labels.shape: {labels.shape}")
                raise
    
    # 분산 전략으로 학습 스텝 실행
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    
    # 손실 값 집계
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


def setup_model_profiling(model, profiler):
    """모델의 모든 레이어에 프로파일러를 설정"""
    def set_profiler_recursive(layer):
        if hasattr(layer, 'set_profiler'):
            layer.set_profiler(profiler)
        
        # 하위 레이어들에도 재귀적으로 적용
        if hasattr(layer, 'layers'):
            for sublayer in layer.layers:
                set_profiler_recursive(sublayer)
        
        # 안전한 속성 확인 - 문제가 될 수 있는 속성들은 제외
        excluded_attrs = {
            'input', 'output', 'input_shape', 'output_shape', 
            'input_spec', 'output_spec', '_input_layers', '_output_layers',
            '_inbound_nodes', '_outbound_nodes', 'built', '_built_input_shape'
        }
        
        # 모든 속성을 확인하여 레이어인 것들에 적용
        for attr_name in dir(layer):
            if (not attr_name.startswith('_') and 
                attr_name not in excluded_attrs):
                try:
                    attr = getattr(layer, attr_name)
                    if (isinstance(attr, tf.keras.layers.Layer) and 
                        hasattr(attr, 'set_profiler')):
                        attr.set_profiler(profiler)
                except (AttributeError, ValueError, RuntimeError):
                    # 레이어가 연결되지 않았거나 다른 이유로 접근할 수 없는 경우 무시
                    continue
    
    set_profiler_recursive(model)
    print(f"🔧 모든 레이어에 프로파일러 설정 완료")


# Whisper 학습 함수 (텐서 사이즈 측정 포함)
def train_whisper_with_profiling(strategy, model_type="small", num_epochs=1, learning_rate=1e-4):
    """Whisper 모델 학습 함수 (고급 텐서 사이즈 측정 포함)"""
    
    # 텐서 프로파일러 초기화
    profiler = TensorProfiler(log_dir='/workspace/tensor_logs')
    
    try:
        with strategy.scope():
            # 모델 생성
            model = create_whisper_model(model_type=model_type)
            
            # 모델을 빌드하기 위해 더미 데이터로 한 번 호출
            dummy_features = tf.random.normal((1, 80, 3000))  # [batch, n_mels, seq_len]
            dummy_labels = tf.random.uniform((1, 100), minval=0, maxval=1000, dtype=tf.int32)
            
            try:
                # 모델 빌드
                _ = model(dummy_features, labels=dummy_labels, training=False)
                print("🔧 모델 빌드 완료")
            except Exception as e:
                print(f"모델 빌드 중 오류 (무시하고 계속): {e}")
            
            # 모델에 프로파일러 설정
            setup_model_profiling(model, profiler)
            
            # 옵티마이저 설정
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            
            # 메트릭 설정
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                tf.keras.metrics.Mean(name="loss")
            ]
            
            # 모델 컴파일
            model.compile(optimizer=optimizer, metrics=metrics)
        
        # 데이터셋 생성
        train_dataset = create_dummy_dataset(GLOBAL_BATCH_SIZE)
        dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
        
        # 체크포인트 설정
        checkpoint_dir = '/workspace/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        
        # 학습 루프
        step = 0
        iterator = iter(dist_dataset)
        
        # 시작 시간 기록
        start_time = time.time()
        
        print(f"=== Tiresias 스타일 텐서 사이즈 측정을 포함한 Whisper-{model_type} 학습 시작 ===")
        
        # 첫 번째 스텝에서 모델 파라미터 로깅
        profiler.start_step(step)
        profiler.log_model_parameters(model)
        profiler.end_step()
        step += 1
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            for batch_idx in range(MAX_ITERATIONS):
                # 분산 데이터셋에서 배치 가져오기
                try:
                    inputs = next(iterator)
                except StopIteration:
                    iterator = iter(dist_dataset)
                    inputs = next(iterator)
                
                # 현재 시간 기록
                step_start = time.time()
                
                # 프로파일링 스텝 시작
                profiler.start_step(step)
                
                # 분산 학습 스텝 실행 (텐서 사이즈 측정 포함)
                loss = distributed_train_step(strategy, model, inputs, optimizer)
                
                # 메모리 사용량 로깅
                memory_info = profiler.log_memory_usage()
                
                # 프로파일링 스텝 종료
                step_tensor_size = profiler.end_step()
                
                # 스텝 완료 시간
                step_end = time.time()
                step_duration = step_end - step_start
                elapsed = step_end - start_time
                
                # 매 10스텝마다 상세 로깅
                if step % 10 == 0:
                    print(f"📊 Step {step} - Loss: {loss.numpy():.4f}")
                    print(f"   💾 GPU Memory: {memory_info['gpu_memory_mb']:.1f} MB, CPU Memory: {memory_info['cpu_memory_mb']:.1f} MB")
                    print(f"   📏 TensorSize: {step_tensor_size:.2f} MB")
                    print(f"   ⏱️  Time: {time.strftime('%H:%M:%S')} (경과: {elapsed:.1f}초, 스텝: {step_duration:.2f}초)")
                else:
                    print(f"Step {step}, Loss: {loss.numpy():.4f}, TensorSize: {step_tensor_size:.2f} MB")
                
                step += 1
            
            # 에포크 종료 후 체크포인트 저장
            checkpoint.save(os.path.join(checkpoint_dir, f"whisper_{model_type}_epoch_{epoch+1}"))
        
        # 최종 결과 저장 및 출력
        print("\n" + "="*60)
        print("🔍 **Tiresias TensorSize 측정 완료**")
        print("="*60)
        
        summary = profiler.save_final_results()
        tiresias_tensorsize = summary['tiresias_tensorsize_mb']
        model_skewness = summary['model_skewness']
        
        print(f"🔍 **Tiresias TensorSize 결과**")
        print(f"whisper_{model_type}    {tiresias_tensorsize:.1f} MB")
        print()
        
        print(f"📊 **모델 Skewness 결과**")
        print(f"whisper_{model_type}    {model_skewness:.1f}")
        print()
        
        # 기존 모델들과 비교표 출력
        reference_models = {
            'alexnet': 6.7,
            'vgg16': 527.8,
            'googlenet': 26.7,
            'inception3': 90.9,
            'resnet50': 97.5,
            'resnet110': 6.6,
            'resnet44': 2.5,
            'resnet56': 3.3,
            'densenet100_k12': 8.5,
            'densenet40_k12': 1.3,
            'bert': 1560,
            'gpt2': 4000
        }
        
        # 레거시 skewness 데이터
        reference_skewness = {
            'alexnet': 2.6,
            'vgg16': 5.1,
            'googlenet': 4.2,
            'inception3': 4.2,
            'resnet50': 3.8,
            'resnet110': 2.3,
            'resnet44': 2.4,
            'resnet56': 2.3,
            'densenet100_k12': 1.9,
            'densenet40_k12': 1.9,
            'bert': 7.3,
            'bertl': 7.2,
            'gpt2': 8,
            'gpt2m': 9.9,
            'gpt2l': 9.8,
            'gpt2xl': 8
        }
        
        print("📊 **모델별 TensorSize 비교** (단위: MB)")
        print("model\t\ttensorsizes")
        for model_name, tensorsize in reference_models.items():
            print(f"{model_name}\t\t{tensorsize}")
        print(f"whisper_{model_type}\t{tiresias_tensorsize:.1f} ⬅️ **이번 측정값**")
        print()
        
        print("📊 **모델별 Skewness 비교**")
        print("model\t\tskewness")
        for model_name, skewness in reference_skewness.items():
            print(f"{model_name}\t\t{skewness}")
        print(f"whisper_{model_type}\t{model_skewness:.1f} ⬅️ **이번 측정값**")
        print()
        
        # 카테고리 분석
        if tiresias_tensorsize < 10:
            category = "경량 모델"
        elif tiresias_tensorsize < 100:
            category = "중간 크기 모델"
        elif tiresias_tensorsize < 1000:
            category = "대형 모델"
        else:
            category = "초대형 모델"
        
        # Skewness 분석
        if model_skewness < 2.0:
            skew_category = "낮은 왜곡도 (균등한 텐서 분포)"
        elif model_skewness < 5.0:
            skew_category = "중간 왜곡도"
        elif model_skewness < 8.0:
            skew_category = "높은 왜곡도"
        else:
            skew_category = "매우 높은 왜곡도 (불균등한 텐서 분포)"
        
        print("📈 **분석 결과:**")
        print(f"- TensorSize 카테고리: {category}")
        print(f"- Skewness 카테고리: {skew_category}")
        
        # 비슷한 크기의 모델 찾기
        closest_models = []
        for model_name, size in reference_models.items():
            if abs(size - tiresias_tensorsize) < tiresias_tensorsize * 0.3:  # 30% 이내
                closest_models.append((model_name, size))
        
        if closest_models:
            closest_names = [name for name, _ in closest_models]
            print(f"- TensorSize 비교: {' ~ '.join(closest_names)} 수준")
        
        # 비슷한 skewness의 모델 찾기
        closest_skew_models = []
        for model_name, skew in reference_skewness.items():
            if abs(skew - model_skewness) < 1.0:  # 1.0 이내
                closest_skew_models.append((model_name, skew))
        
        if closest_skew_models:
            closest_skew_names = [name for name, _ in closest_skew_models]
            print(f"- Skewness 비교: {' ~ '.join(closest_skew_names)} 수준")
        
        print(f"- 한 iteration당 처리하는 텐서 총 크기: {tiresias_tensorsize:.1f} MB")
        print(f"- 텐서 크기 분포의 왜곡도: {model_skewness:.1f}")
        print()
        
        print("💡 **지표 의미:**")
        print("- TensorSize: 한 번의 학습 iteration에서 처리되는 모든 텐서의 총 메모리 크기(MB)")
        print("- Skewness: 텐서 크기 분포의 비대칭성 (0에 가까울수록 균등한 분포)")
        print("  * 양수: 큰 텐서가 적고 작은 텐서가 많음")
        print("  * 음수: 작은 텐서가 적고 큰 텐서가 많음")
        print("  * 절댓값이 클수록: 더 불균등한 분포")
        print("- GPU 메모리 요구량 예측과 작업 스케줄링 최적화에 사용")
        print("="*60)
        
        return model, summary
        
    finally:
        # 프로파일러 종료
        profiler.close()


# 텐서 사이즈 로깅 함수
def log_tensor_sizes(step, save_dir):
    """텐서 사이즈를 로깅하고 파일에 저장"""
    summary = tensor_monitor.get_step_summary()
    step_size_mb = summary['step_total_size'] / (1024 * 1024)  # MB 단위로 변환
    
    print(f"Step {step} - Total tensor size: {step_size_mb:.2f} MB")
    
    # 상위 10개 operation별 텐서 사이즈 출력
    op_sizes = summary['operation_sizes']
    sorted_ops = sorted(op_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print("Top 10 operations by tensor size:")
    for op_name, size in sorted_ops:
        size_mb = size / (1024 * 1024)
        print(f"  {op_name}: {size_mb:.2f} MB")
    
    # 파일에 저장
    os.makedirs(save_dir, exist_ok=True)
    
    # 스텝별 총 텐서 사이즈 저장
    with open(os.path.join(save_dir, 'step_tensor_sizes.txt'), 'a') as f:
        f.write(f"{step},{step_size_mb:.2f}\n")
    
    # Operation별 텐서 사이즈 저장 (JSON 형태)
    tensor_log = {
        'step': step,
        'total_size_mb': step_size_mb,
        'operation_sizes': {op: size/(1024*1024) for op, size in op_sizes.items()}
    }
    
    with open(os.path.join(save_dir, f'tensor_sizes_step_{step}.json'), 'w') as f:
        json.dump(tensor_log, f, indent=2)


# Whisper 모델 생성 함수
def create_whisper_model(model_type="small"):
    """
    지정된 크기의 Whisper 모델 생성
    """
    config = WhisperConfig()
    
    # 모델 크기에 따른 설정 조정
    if model_type == "tiny":
        config.d_model = 384
        config.encoder_layers = 4
        config.encoder_attention_heads = 6
        config.decoder_layers = 4
        config.decoder_attention_heads = 6
        config.d_ff = 1536
    elif model_type == "base":
        config.d_model = 512
        config.encoder_layers = 6
        config.encoder_attention_heads = 8
        config.decoder_layers = 6
        config.decoder_attention_heads = 8
        config.d_ff = 2048
    elif model_type == "medium":
        config.d_model = 1024
        config.encoder_layers = 24
        config.encoder_attention_heads = 16
        config.decoder_layers = 24
        config.decoder_attention_heads = 16
        config.d_ff = 4096
    elif model_type == "large":
        config.d_model = 1280
        config.encoder_layers = 32
        config.encoder_attention_heads = 20
        config.decoder_layers = 32
        config.decoder_attention_heads = 20
        config.d_ff = 5120
    
    # Whisper-small은 기본값 사용 (config 생성 시 이미 설정됨)
    
    return WhisperForConditionalGeneration(config)


# 추론 함수
def transcribe_audio(model, audio_path, tokenizer=None, max_length=448):
    """
    오디오 파일에서 텍스트 추출
    """
    # 오디오 로드 및 전처리
    # 실제 구현에서는 여기에 오디오 로드 코드가 필요함
    # 여기서는 더미 데이터 사용
    dummy_waveform = np.random.randn(16000 * 30).astype(np.float32)  # 30초 오디오
    
    # 특징 추출
    features = extract_fbank_features(dummy_waveform)
    features = tf.expand_dims(features, 0)  # 배치 차원 추가
    
    # 추론
    decoder_input_ids = tf.fill((1, 1), model.config.decoder_start_token_id)
    outputs = model.generate(features, max_length=max_length)
    
    # 토큰 ID에서 텍스트 변환
    if tokenizer is not None:
        transcription = tokenizer.decode(outputs[0].numpy())
    else:
        # 토크나이저가 없는 경우 ID만 반환
        transcription = outputs[0].numpy()
    
    return transcription


# 메인 함수
def main(strategy):
    print("Whisper-small 분산 학습 시작 (Tiresias 스타일 텐서 사이즈 측정 포함)...")
    
    # 네트워크 및 GPU 모니터링 시작
    os.system('sh /workspace/network.sh &')  # network profile
    os.system('sh /workspace/gpu.sh &')  # gpu profile
    print('''
========================
network profile started!
Tiresias-style tensor size monitoring enabled!
========================''')
    
    # JCT 측정 시작
    start_time = time.time()
    
    # 모델 학습 실행 (Tiresias 스타일 텐서 사이즈 측정 포함)
    model, tensor_summary = train_whisper_with_profiling(strategy, model_type="small")
    
    # JCT 측정 종료
    end_time = time.time()
    jct = end_time - start_time
    
    # 결과 출력
    print("Training completed.")
    print("jct:", jct)
    
    # JCT 파일 저장
    try:
        model_txt = open('/workspace/model.txt', 'r')
        save_dir_name = model_txt.read()
        result_dir = '/result/' + save_dir_name.strip()
        os.makedirs(result_dir, exist_ok=True)
        
        jct_file = open(result_dir + '/' + task_type + '_' + str(task_index) + '_jct.txt', 'w')
        jct_file.write('%.2f' % (float(jct)))
        jct_file.close()
        model_txt.close()
        
        # 텐서 사이즈 로그를 결과 디렉토리에 복사
        tensor_log_source = '/workspace/tensor_logs'
        tensor_log_dest = result_dir + '/tensor_logs'
        
        try:
            import shutil
            if os.path.exists(tensor_log_source):
                shutil.copytree(tensor_log_source, tensor_log_dest, dirs_exist_ok=True)
                print(f"🔍 텐서 사이즈 로그가 {tensor_log_dest}에 저장되었습니다.")
        except Exception as e:
            print(f"텐서 로그 복사 중 오류 발생: {e}")
        
        # Tiresias 결과를 별도 파일에 저장
        tiresias_result_file = result_dir + '/tiresias_tensorsize_result.txt'
        with open(tiresias_result_file, 'w') as f:
            f.write(f"model,tensorsize_mb\n")
            f.write(f"whisper_small,{tensor_summary['tiresias_tensorsize_mb']:.1f}\n")
        print(f"🔍 Tiresias 결과가 {tiresias_result_file}에 저장되었습니다.")
        
        # Skewness 결과를 별도 파일에 저장 (레거시 포맷)
        skewness_result_file = result_dir + '/legacy_skewness_result.txt'
        with open(skewness_result_file, 'w') as f:
            f.write(f"model,skewness\n")
            f.write(f"whisper_small,{tensor_summary['model_skewness']:.1f}\n")
        print(f"📊 Skewness 결과가 {skewness_result_file}에 저장되었습니다.")
        
        # 통합 결과 파일 저장
        combined_result_file = result_dir + '/combined_metrics_result.txt'
        with open(combined_result_file, 'w') as f:
            f.write(f"model,tensorsize_mb,skewness\n")
            f.write(f"whisper_small,{tensor_summary['tiresias_tensorsize_mb']:.1f},{tensor_summary['model_skewness']:.1f}\n")
        print(f"🔗 통합 결과가 {combined_result_file}에 저장되었습니다.")
        
    except Exception as e:
        print(f"결과 저장 중 오류 발생: {e}")
    
    # 모델 저장
    model_path = os.path.join(CACHE_DIR, "whisper_small_model")
    try:
        model.save_weights(model_path)
        print(f"모델이 {model_path}에 저장되었습니다.")
    except Exception as e:
        print(f"모델 저장 중 오류 발생: {e}")
    
    print("\n🎉 **Whisper 텐서 분석 완료!**")
    print(f"📊 최종 TensorSize: {tensor_summary['tiresias_tensorsize_mb']:.1f} MB")
    print(f"📊 최종 Skewness: {tensor_summary['model_skewness']:.1f}")
    print("🔍 상세 로그는 /workspace/tensor_logs 디렉토리에서 확인하세요.")


if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Whisper-small Distributed Speech Recognition with Tensor Size Monitoring')
    parser.add_argument('--num_batches', type=int, default=40, help='num_batches per replica, default is set 40')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size per replica, default is set 1')
    parser.add_argument('--log_tensor_freq', type=int, default=1, help='frequency of tensor size logging (every N steps), default is 1')
    args = parser.parse_args()

    # 환경 설정
    tf_config = json.loads(os.environ.get('TF_CONFIG') or '{}')
    task_config = tf_config.get('task', {})
    task_type = task_config.get('type')
    task_index = task_config.get('index')

    # 모델과 데이터셋을 저장할 로컬 디렉토리 설정
    CACHE_DIR = '/workspace/model_cache'  # 컨테이너 내 사전 준비된 모델 캐시 경로
    DATASET_DIR = '/workspace/datasets'  # 컨테이너 내 사전 준비된 데이터셋 경로

    # 분산 학습 전략 설정
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # 하이퍼파라미터 설정
    BATCH_SIZE_PER_REPLICA = args.batch_size
    GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    MAX_ITERATIONS = args.num_batches
    BUFFER_SIZE = 10000
    TENSOR_LOG_FREQ = args.log_tensor_freq

    print(f'batch size per replica: {BATCH_SIZE_PER_REPLICA}, global batch size: {GLOBAL_BATCH_SIZE}')
    print(f'num_batches: {MAX_ITERATIONS}')
    print(f'tensor logging frequency: every {TENSOR_LOG_FREQ} steps')
    
    main(strategy)