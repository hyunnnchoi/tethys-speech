# Tiny 모델 (~15-20M 파라미터):
# hidden_size: 256
# num_hidden_layers: 4
# num_attention_heads: 4
# intermediate_size: 1024
# conv_dim: [128, 128, 128, 128] (4 layers)

# Small 모델 (~30-40M 파라미터):
# hidden_size: 512
# num_hidden_layers: 6
# num_attention_heads: 8
# intermediate_size: 2048
# conv_dim: [256, 256, 256, 256, 256] (5 layers)

import tensorflow as tf
import numpy as np
import json
import os
import sys
import time
import argparse


class Wav2Vec2Config:
    def __init__(self, model_size="small"):
        # 모델 크기에 따른 설정
        if model_size == "small":
            # Small 모델 설정 (약 30-40M 파라미터)
            self.hidden_size = 512
            self.num_hidden_layers = 6
            self.num_attention_heads = 8
            self.intermediate_size = 2048
            self.conv_dim = [256, 256, 256, 256, 256]  # 5 layers, 채널 수 줄임
            self.conv_stride = [5, 2, 2, 2, 2]
            self.conv_kernel = [10, 3, 3, 3, 2]
            self.num_conv_pos_embeddings = 64  # 128 → 64
            self.num_conv_pos_embedding_groups = 8  # 16 → 8
            
        elif model_size == "tiny":
            # Tiny 모델 설정 (약 15-20M 파라미터)
            self.hidden_size = 256
            self.num_hidden_layers = 4
            self.num_attention_heads = 4
            self.intermediate_size = 1024
            self.conv_dim = [128, 128, 128, 128]  # 4 layers
            self.conv_stride = [5, 2, 2, 2]
            self.conv_kernel = [10, 3, 3, 2]
            self.num_conv_pos_embeddings = 32
            self.num_conv_pos_embedding_groups = 4
            
        else:  # base 모델 (기존 설정)
            self.hidden_size = 768
            self.num_hidden_layers = 12
            self.num_attention_heads = 12
            self.intermediate_size = 3072
            self.conv_dim = [512, 512, 512, 512, 512, 512, 512]
            self.conv_stride = [5, 2, 2, 2, 2, 2, 2]
            self.conv_kernel = [10, 3, 3, 3, 3, 2, 2]
            self.num_conv_pos_embeddings = 128
            self.num_conv_pos_embedding_groups = 16

        # 특징 추출기 설정
        self.feat_extract_norm = "group"
        self.feat_extract_activation = "gelu"
        self.conv_bias = False
        
        # 공통 설정들
        self.hidden_act = "gelu"
        self.hidden_dropout = 0.1
        self.activation_dropout = 0.1
        self.attention_dropout = 0.1
        self.layer_norm_eps = 1e-5
        
        # 양자화 및 마스킹 설정 (small 모델에 맞게 조정)
        if model_size == "small":
            self.num_codevectors_per_group = 160  # 320 → 160
            self.num_codevector_groups = 2
            self.codevector_dim = 128  # 256 → 128
            self.proj_codevector_dim = 128  # 256 → 128
        elif model_size == "tiny":
            self.num_codevectors_per_group = 80
            self.num_codevector_groups = 2
            self.codevector_dim = 64
            self.proj_codevector_dim = 64
        else:  # base
            self.num_codevectors_per_group = 320
            self.num_codevector_groups = 2
            self.codevector_dim = 256
            self.proj_codevector_dim = 256
            
        self.contrastive_logits_temperature = 0.1
        self.num_negatives = 100
        self.diversity_loss_weight = 0.1
        self.ctc_loss_reduction = "sum"
        self.ctc_zero_infinity = False
        
        # 미세 조정 설정
        self.mask_time_prob = 0.05
        self.mask_time_length = 10
        self.mask_feature_prob = 0.0
        self.mask_feature_length = 10
        
        # 추가 설정
        self.vocab_size = 32
        self.do_stable_layer_norm = True
        self.use_weighted_layer_sum = False
        
        # 분류 관련 설정 (small 모델에 맞게 조정)
        if model_size == "small":
            self.classifier_proj_size = 128  # 256 → 128
        elif model_size == "tiny":
            self.classifier_proj_size = 64
        else:
            self.classifier_proj_size = 256
            
        # TDNN 설정도 작게 조정
        if model_size == "small":
            self.tdnn_dim = [256, 256, 256, 256, 768]
            self.xvector_output_dim = 256
        elif model_size == "tiny":
            self.tdnn_dim = [128, 128, 128, 128, 384]
            self.xvector_output_dim = 128
        else:
            self.tdnn_dim = [512, 512, 512, 512, 1500]
            self.xvector_output_dim = 512
            
        self.tdnn_kernel = [5, 3, 3, 1, 1]
        self.tdnn_dilation = [1, 2, 3, 1, 1]
        
        # 분류를 위한 설정
        self.num_labels = 10


# Gelu 활성화 함수 정의
def gelu(x):
    """
    Gaussian Error Linear Unit 활성화 함수
    """
    return 0.5 * x * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))


# 그룹 정규화 레이어
class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self, groups=32, axis=-1, epsilon=1e-5):
        super(GroupNormalization, self).__init__()
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
    
    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Dimension of axis {} must be known'.format(self.axis))
        if dim % self.groups != 0:
            raise ValueError('Dimension of axis {} must be divisible by groups {}'.format(
                self.axis, self.groups))
        
        self.gamma = self.add_weight(
            name='gamma',
            shape=(dim,),
            initializer='ones',
            trainable=True)
        
        self.beta = self.add_weight(
            name='beta',
            shape=(dim,),
            initializer='zeros',
            trainable=True)
    
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        reshaped_inputs = self._reshape_into_groups(inputs)
        
        normalized_inputs = self._apply_normalization(reshaped_inputs, input_shape)
        
        return tf.reshape(normalized_inputs, input_shape)
    
    def _reshape_into_groups(self, inputs):
        input_shape = tf.shape(inputs)
        group_shape = [input_shape[0], input_shape[1], self.groups, input_shape[2] // self.groups]
        
        # [batch, time, features] -> [batch, time, groups, features_per_group]
        group_shaped_inputs = tf.reshape(inputs, group_shape)
        
        # [batch, time, groups, features_per_group] -> [batch, time, features_per_group, groups]
        return tf.transpose(group_shaped_inputs, [0, 1, 3, 2])
    
    def _apply_normalization(self, reshaped_inputs, input_shape):
        # 평균과 분산 계산
        mean, variance = tf.nn.moments(reshaped_inputs, [1, 2], keepdims=True)
        # 정규화
        normalized_inputs = (reshaped_inputs - mean) / tf.sqrt(variance + self.epsilon)
        
        # 원래 형태로 복원
        normalized_inputs = tf.transpose(normalized_inputs, [0, 1, 3, 2])
        normalized_inputs = tf.reshape(normalized_inputs, input_shape)
        
        # 스케일 및 시프트 적용
        return self.gamma * normalized_inputs + self.beta


# 상대적 위치 임베딩 레이어
class RelativePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, dim, max_length=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.dim = dim
        pe = self._get_pos_encoding(max_length, dim)
        self.pos_embedding = tf.Variable(
            initial_value=pe,
            trainable=True,
            dtype=tf.float32,
            name="pos_embedding"
        )
    
    def _get_pos_encoding(self, max_length, d_model):
        # 위치 임베딩 초기화
        pos_enc = np.zeros((max_length, d_model))
        for pos in range(max_length):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        return pos_enc
    
    def call(self, length):
        return self.pos_embedding[:length] 


# ============ 기본 모델 컴포넌트 ============ #

# 특징 추출기 (Feature Extractor) 구현
class Wav2Vec2FeatureExtractor(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Wav2Vec2FeatureExtractor, self).__init__()
        self.config = config
        
        # 컨볼루션 레이어 스택 생성
        self.conv_layers = []
        
        # 첫 번째 컨볼루션 레이어
        self.conv_layers.append(
            tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    filters=config.conv_dim[0],
                    kernel_size=config.conv_kernel[0],
                    strides=config.conv_stride[0],
                    padding="same",
                    use_bias=config.conv_bias,
                    name="conv_0"
                ),
                GroupNormalization(groups=self.config.num_conv_pos_embedding_groups) if config.feat_extract_norm == "group" else tf.keras.layers.LayerNormalization(epsilon=1e-5),
                tf.keras.layers.Activation(gelu if config.feat_extract_activation == "gelu" else "relu")
            ])
        )
        
        # 나머지 컨볼루션 레이어 추가
        for i in range(1, len(config.conv_dim)):
            self.conv_layers.append(
                tf.keras.Sequential([
                    tf.keras.layers.Conv1D(
                        filters=config.conv_dim[i],
                        kernel_size=config.conv_kernel[i],
                        strides=config.conv_stride[i],
                        padding="same",
                        use_bias=config.conv_bias,
                        name=f"conv_{i}"
                    ),
                    GroupNormalization(groups=self.config.num_conv_pos_embedding_groups) if config.feat_extract_norm == "group" else tf.keras.layers.LayerNormalization(epsilon=1e-5),
                    tf.keras.layers.Activation(gelu if config.feat_extract_activation == "gelu" else "relu")
                ])
            )
        
        # 위치 인코딩 레이어
        self.pos_conv_embed = tf.keras.layers.Conv1D(
            filters=config.conv_dim[-1],
            kernel_size=config.num_conv_pos_embeddings,
            padding="same",
            groups=config.num_conv_pos_embedding_groups,
            name="pos_conv_embed"
        )
        
        # 레이어 정규화
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
    
    def call(self, inputs, training=False):
        hidden_states = tf.expand_dims(inputs, axis=-1)  # [batch, time] -> [batch, time, 1]
        
        # 컨볼루션 레이어 통과
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states, training=training)
        
        # 위치 인코딩 추가
        position_embeddings = self.pos_conv_embed(hidden_states)
        
        # 합산 및 정규화
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        
        return hidden_states


# MultiHeadAttention 구현
class Wav2Vec2MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Wav2Vec2MultiHeadAttention, self).__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
            )
        
        # 어텐션 프로젝션 레이어
        self.q_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True, name="q_proj")
        self.k_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True, name="k_proj")
        self.v_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(self.embed_dim, use_bias=True, name="out_proj")
        
        self.dropout = tf.keras.layers.Dropout(config.attention_dropout)
    
    def _reshape_for_multihead_attention(self, x):
        batch_size = tf.shape(x)[0]
        seq_length = tf.shape(x)[1]
        
        # [batch, time, embed_dim] -> [batch, time, num_heads, head_dim]
        reshaped = tf.reshape(x, (batch_size, seq_length, self.num_heads, self.head_dim))
        
        # [batch, time, num_heads, head_dim] -> [batch, num_heads, time, head_dim]
        return tf.transpose(reshaped, perm=[0, 2, 1, 3])
    
    def call(self, hidden_states, attention_mask=None, output_attentions=False, training=False):
        batch_size = tf.shape(hidden_states)[0]
        seq_length = tf.shape(hidden_states)[1]
        
        # 선형 투영
        query_states = self.q_proj(hidden_states)  # [batch, time, embed_dim]
        key_states = self.k_proj(hidden_states)    # [batch, time, embed_dim]
        value_states = self.v_proj(hidden_states)  # [batch, time, embed_dim]
        
        # 다중 헤드 형태로 변환
        query_states = self._reshape_for_multihead_attention(query_states)  # [batch, num_heads, time, head_dim]
        key_states = self._reshape_for_multihead_attention(key_states)      # [batch, num_heads, time, head_dim]
        value_states = self._reshape_for_multihead_attention(value_states)  # [batch, num_heads, time, head_dim]
        
        # QK^T 계산
        attention_scores = tf.matmul(query_states, key_states, transpose_b=True)  # [batch, num_heads, time, time]
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # 어텐션 마스크 적용 (필요한 경우)
        if attention_mask is not None:
            # 마스크를 [batch, 1, 1, time] 형태로 변환하여 브로드캐스팅 활용
            attention_mask = tf.expand_dims(tf.expand_dims(attention_mask, 1), 1)
            attention_scores = attention_scores + (1.0 - attention_mask) * -10000.0
        
        # 소프트맥스 적용
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        
        # 값과 어텐션 가중치 결합
        context_states = tf.matmul(attention_probs, value_states)  # [batch, num_heads, time, head_dim]
        
        # 형태 변환: [batch, num_heads, time, head_dim] -> [batch, time, num_heads, head_dim]
        context_states = tf.transpose(context_states, perm=[0, 2, 1, 3])
        
        # 최종 형태로 변환: [batch, time, embed_dim]
        context_states = tf.reshape(context_states, (batch_size, seq_length, self.embed_dim))
        
        # 최종 프로젝션
        output = self.out_proj(context_states)
        
        # 어텐션 가중치 반환 (필요한 경우)
        outputs = (output, attention_probs) if output_attentions else (output,)
        
        return outputs


# FeedForward 네트워크 구현
class Wav2Vec2FeedForward(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Wav2Vec2FeedForward, self).__init__()
        self.intermediate_dense = tf.keras.layers.Dense(config.intermediate_size, name="intermediate_dense")
        self.intermediate_act_fn = gelu if config.hidden_act == "gelu" else tf.keras.activations.get(config.hidden_act)
        self.intermediate_dropout = tf.keras.layers.Dropout(config.activation_dropout)
        
        self.output_dense = tf.keras.layers.Dense(config.hidden_size, name="output_dense")
        self.output_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
    
    def call(self, hidden_states, training=False):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states, training=training)
        
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states, training=training)
        
        return hidden_states 


# Transformer 인코더 레이어 구현
class Wav2Vec2EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Wav2Vec2EncoderLayer, self).__init__()
        self.config = config
        self.do_stable_layer_norm = config.do_stable_layer_norm
        
        # 어텐션 레이어
        self.attention = Wav2Vec2MultiHeadAttention(config)
        self.attention_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.attention_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        
        # 피드포워드 레이어
        self.feed_forward = Wav2Vec2FeedForward(config)
        self.feed_forward_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
    
    def call(self, hidden_states, attention_mask=None, output_attentions=False, training=False):
        # self-attention 레이어
        if self.do_stable_layer_norm:
            # 레이어 정규화 후 어텐션
            attention_input = self.attention_layer_norm(hidden_states)
            attention_outputs = self.attention(
                attention_input,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training
            )
            attention_output = attention_outputs[0]
            
            # 잔차 연결
            attention_output = self.attention_dropout(attention_output, training=training)
            hidden_states = hidden_states + attention_output
            
            # 레이어 정규화 후 피드포워드
            feed_forward_input = self.feed_forward_layer_norm(hidden_states)
            feed_forward_output = self.feed_forward(feed_forward_input, training=training)
            
            # 잔차 연결
            hidden_states = hidden_states + feed_forward_output
        else:
            # 기존 방식: 어텐션 후 레이어 정규화
            attention_outputs = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                training=training
            )
            attention_output = attention_outputs[0]
            
            # 잔차 연결 및 레이어 정규화
            attention_output = self.attention_dropout(attention_output, training=training)
            hidden_states = self.attention_layer_norm(hidden_states + attention_output)
            
            # 피드포워드 및 정규화
            feed_forward_output = self.feed_forward(hidden_states, training=training)
            hidden_states = self.feed_forward_layer_norm(hidden_states + feed_forward_output)
        
        outputs = (hidden_states,) + attention_outputs[1:] if output_attentions else (hidden_states,)
        
        return outputs


# Transformer 인코더 구현
class Wav2Vec2Encoder(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Wav2Vec2Encoder, self).__init__()
        self.config = config
        self.layers = [Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.gradient_checkpointing = False
        
        # 가중치 합계를 위한 레이어 (필요한 경우)
        self.layer_weights = None
        if config.use_weighted_layer_sum:
            self.layer_weights = tf.Variable(
                initial_value=tf.ones(config.num_hidden_layers) / config.num_hidden_layers,
                trainable=True,
                dtype=tf.float32,
                name="layer_weights"
            )
    
    def call(self, 
             hidden_states, 
             attention_mask=None,
             output_hidden_states=False, 
             output_attentions=False, 
             return_dict=True,
             training=False):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        if self.config.use_weighted_layer_sum:
            # 0번째 레이어 출력을 초기값으로 사용
            layerwise_hidden_states = ()
            for i, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                
                # 정규화된 가중치 계산
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    training=training
                )
                hidden_states = layer_outputs[0]
                
                layerwise_hidden_states = layerwise_hidden_states + (hidden_states,)
                
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            
            # 가중치 정규화
            norm_weights = tf.nn.softmax(self.layer_weights, axis=-1)
            
            # 정규화된 가중치로 각 레이어의 출력 가중합 계산
            hidden_states = tf.stack(layerwise_hidden_states, axis=0)  # [num_layers, batch, time, hidden_size]
            hidden_states = tf.einsum("l,lbth->bth", norm_weights, hidden_states)
        else:
            # 기본 구현: 각 레이어 순차적으로 통과
            for i, layer in enumerate(self.layers):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    output_attentions=output_attentions,
                    training=training
                )
                hidden_states = layer_outputs[0]
                
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
        
        # 마지막 레이어 출력도 추가
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        
        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions
        }


# 프로젝션 헤드 구현
class Wav2Vec2ProjectionHead(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Wav2Vec2ProjectionHead, self).__init__()
        self.dense = tf.keras.layers.Dense(config.proj_codevector_dim, name="projection_head")
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
    
    def call(self, hidden_states, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        return hidden_states 


# Wav2Vec2Quantizer 클래스 구현 (수정됨)
class Wav2Vec2Quantizer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(Wav2Vec2Quantizer, self).__init__()
        self.config = config
        
        # 코드북 벡터 초기화
        self.codevectors = tf.Variable(
            initial_value=tf.random.normal(
                [config.num_codevector_groups, config.num_codevectors_per_group, config.codevector_dim // config.num_codevector_groups]
            ),
            trainable=True,
            name="codevectors"
        )
        
        # 입력 차원을 codevector_dim으로 투영하는 레이어
        self.projection = tf.keras.layers.Dense(config.codevector_dim, name="project_to_codevector_dim")
    
    def call(self, hidden_states, training=False):
        original_shape = tf.shape(hidden_states)
        batch_size, sequence_length = original_shape[0], original_shape[1]
        
        # 입력을 codevector_dim으로 투영
        hidden_states = self.projection(hidden_states)
        
        # 실제 배치 크기가 0인 경우 처리 (분산 학습에서 발생 가능)
        if tf.equal(batch_size, 0):
            # 빈 텐서 반환
            quantized_features = tf.zeros([0, sequence_length, self.config.codevector_dim], dtype=tf.float32)
            dummy_encodings = tf.zeros([self.config.num_codevector_groups, 0, sequence_length, self.config.num_codevectors_per_group], dtype=tf.float32)
            dummy_distances = tf.zeros([self.config.num_codevector_groups, 0, sequence_length, self.config.num_codevectors_per_group], dtype=tf.float32)
            
            return {
                "quantized_features": quantized_features,
                "encodings": dummy_encodings,
                "distances": dummy_distances,
                "codevector_perplexity": tf.constant(1.0)
            }
        
        # hidden_states를 그룹 수에 맞게 재구성
        group_dim = self.config.codevector_dim // self.config.num_codevector_groups
        hidden_states = tf.reshape(
            hidden_states, 
            [batch_size, sequence_length, self.config.num_codevector_groups, group_dim]
        )
        
        # 각 그룹별로 코드벡터와의 거리 계산
        distances = []
        quantized_features = []
        encodings = []
        
        for i in range(self.config.num_codevector_groups):
            group_hidden = hidden_states[:, :, i, :]  # [batch, time, group_dim]
            
            # 현재 그룹의 코드벡터들 [num_vectors, group_dim]
            group_vectors = self.codevectors[i]
            
            # 거리 계산을 위해 텐서를 확장
            # group_hidden: [batch, time, 1, group_dim]
            # group_vectors: [1, 1, num_vectors, group_dim]
            expanded_hidden = tf.expand_dims(group_hidden, axis=2)
            expanded_vectors = tf.expand_dims(tf.expand_dims(group_vectors, axis=0), axis=0)
            
            # 유클리드 거리 계산: [batch, time, num_vectors]
            dist = tf.reduce_sum(tf.square(expanded_hidden - expanded_vectors), axis=-1)
            distances.append(dist)
            
            # 가장 가까운 코드벡터 인덱스: [batch, time]
            indices = tf.argmin(dist, axis=-1)
            
            # one-hot 인코딩: [batch, time, num_vectors]
            encoding = tf.one_hot(indices, self.config.num_codevectors_per_group, dtype=tf.float32)
            encodings.append(encoding)
            
            # 양자화된 특징: [batch, time, group_dim]
            quantized = tf.matmul(encoding, group_vectors)
            quantized_features.append(quantized)
        
        # 결과 결합
        # distances: [num_groups, batch, time, num_vectors]
        distances = tf.stack(distances, axis=0)
        
        # encodings: [num_groups, batch, time, num_vectors]  
        encodings = tf.stack(encodings, axis=0)
        
        # quantized_features: [batch, time, codevector_dim]
        quantized_features = tf.concat(quantized_features, axis=-1)
        
        # 다양성 계산 (코드북 활용도)
        # 시간 차원에서 평균을 계산할 때 실제 길이 고려
        avg_probs = tf.reduce_mean(encodings, axis=[1, 2])  # [num_groups, num_vectors]
        
        # 수치적 안정성을 위한 작은 값 추가
        avg_probs = tf.clip_by_value(avg_probs, 1e-10, 1.0)
        
        # 퍼플렉시티 계산
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10), axis=-1))
        perplexity = tf.reduce_mean(perplexity)
        
        return {
            "quantized_features": quantized_features,
            "encodings": encodings,
            "distances": distances,
            "codevector_perplexity": perplexity
        } 


# ============ 주요 모델 클래스 ============ #

# 메인 wav2vec2 모델 구현
class Wav2Vec2Model(tf.keras.Model):
    def __init__(self, config):
        super(Wav2Vec2Model, self).__init__()
        self.config = config
        
        # 특징 추출기 및 투영
        self.feature_extractor = Wav2Vec2FeatureExtractor(config)
        self.feature_projection = tf.keras.layers.Dense(config.hidden_size, name="feature_projection")
        self.feature_projection_layer_norm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps)
        self.feature_projection_dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        
        # 인코더
        self.encoder = Wav2Vec2Encoder(config)
        
        # 양자화기 (pre-training에서 사용)
        self.quantizer = Wav2Vec2Quantizer(config)

        # 투영 헤드
        self.project_hid = Wav2Vec2ProjectionHead(config)
        self.project_q = Wav2Vec2ProjectionHead(config)
    
    def call(self, inputs, attention_mask=None, 
             output_attentions=False, output_hidden_states=False,
             return_dict=True, training=False):
        
        # 1. 특징 추출
        extract_features = self.feature_extractor(inputs, training=training)
        
        # 2. 특징 투영 - hidden_size 차원으로 투영
        # 특징 추출기의 출력 차원(conv_dim[-1])과 인코더의 입력 차원(hidden_size)을 맞추기 위한 투영
        hidden_states = self.feature_projection(extract_features)
        hidden_states = self.feature_projection_layer_norm(hidden_states)
        hidden_states = self.feature_projection_dropout(hidden_states, training=training)
        
        # 3. 타겟 양자화를 위한 특징 제공
        if training:
            # 양자화 대상 특징 - 투영된 특징을 사용
            quantize_targets = hidden_states
            
            # 양자화
            quantized_result = self.quantizer(quantize_targets, training=True)
            quantized_features = quantized_result["quantized_features"]
            codevector_perplexity = quantized_result["codevector_perplexity"]
        else:
            quantized_features = None
            codevector_perplexity = None
        
        # 4. 인코더에 전달
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training
        )
        
        if not return_dict:
            return encoder_outputs
        
        # 반환값 준비
        result = {
            "last_hidden_state": encoder_outputs["last_hidden_state"],
            "extract_features": extract_features
        }
        
        if quantized_features is not None:
            result["quantized_features"] = quantized_features
        
        if codevector_perplexity is not None:
            result["codevector_perplexity"] = codevector_perplexity
        
        if output_hidden_states:
            result["hidden_states"] = encoder_outputs["hidden_states"]
        
        if output_attentions:
            result["attentions"] = encoder_outputs["attentions"]
        
        return result


# 사전학습 wav2vec2 모델 구현 (수정됨)
class Wav2Vec2ForPreTraining(tf.keras.Model):
    def __init__(self, config):
        super(Wav2Vec2ForPreTraining, self).__init__()
        self.config = config
        
        # 기본 wav2vec2 모델
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # 손실 계산 및 학습에 필요한 매개변수
        self.num_negatives = config.num_negatives
        self.contrastive_logits_temperature = config.contrastive_logits_temperature
        self.diversity_loss_weight = config.diversity_loss_weight
    
    def call(self, inputs, attention_mask=None, output_attentions=False, output_hidden_states=False, training=False):
        # wav2vec2 모델 호출
        outputs = self.wav2vec2(
            inputs,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            training=training
        )
        
        # 컨트라스트 학습을 위한 변환
        if training and "quantized_features" in outputs:
            # 인코더 출력 프로젝션
            transformer_features = self.wav2vec2.project_hid(outputs["last_hidden_state"])
            
            # 양자화된 특징 프로젝션
            quantized_features = self.wav2vec2.project_q(outputs["quantized_features"])
            
            # 손실 계산을 위한 정보 추가
            outputs["projected_quantized_features"] = quantized_features
            outputs["projected_states"] = transformer_features
        
        return outputs
    
    @tf.function
    def _compute_contrastive_loss(self, hidden_states, quantized_states):
        """개선된 컨트라스트 손실 계산"""
        batch_size = tf.shape(hidden_states)[0]
        sequence_length = tf.shape(hidden_states)[1]
        
        # 빈 배치 처리
        if tf.equal(batch_size, 0):
            return tf.zeros([0, 0, 1], dtype=tf.float32), tf.constant(0.0)
        
        # 양수 쌍 계산
        pos_logits = tf.reduce_sum(hidden_states * quantized_states, axis=-1) / self.contrastive_logits_temperature
        
        # 음수 쌍 생성 (개선된 방법)
        if self.num_negatives > 0:
            # 더 안전한 negative sampling
            neg_indices = self._sample_negative_indices(sequence_length, batch_size)
            
            # 안전한 gather 연산
            neg_quantized = tf.gather(quantized_states, neg_indices, axis=1, batch_dims=1)
            
            # 차원 확장 및 로짓 계산
            hidden_states_expanded = tf.expand_dims(hidden_states, axis=2)
            neg_logits = tf.reduce_sum(hidden_states_expanded * neg_quantized, axis=-1) / self.contrastive_logits_temperature
            
            # 로짓 결합
            logits = tf.concat([tf.expand_dims(pos_logits, axis=2), neg_logits], axis=2)
        else:
            logits = tf.expand_dims(pos_logits, axis=2)
        
        # 손실 계산
        labels = tf.zeros([batch_size, sequence_length], dtype=tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        
        return logits, tf.reduce_mean(loss)
    
    def _compute_diversity_loss(self, perplexity):
        """다양성 손실 계산 - 코드북 활용도를 높이기 위함"""
        # 퍼플렉시티가 높을수록 다양성이 높음을 의미
        # 인위적으로 다양성을 높이기 위해 -perplexity를 최소화
        return -perplexity
    
    @tf.function  
    def _sample_negative_indices(self, sequence_length, batch_size):
        """안전한 negative sampling - TensorFlow 그래프 호환"""
        # sequence_length가 num_negatives보다 작은 경우 처리
        actual_negatives = tf.minimum(self.num_negatives, sequence_length - 1)
        actual_negatives = tf.maximum(actual_negatives, 1)  # 최소 1개는 보장
        
        # 배치별로 negative 인덱스 생성 (벡터화된 방식)
        indices_range = tf.range(sequence_length)
        
        # 각 배치에 대해 독립적으로 셔플
        # tf.random.shuffle은 배치 차원에서 작동하지 않으므로 다른 방법 사용
        random_indices = tf.random.uniform([batch_size, sequence_length], maxval=sequence_length, dtype=tf.int32)
        
        # 상위 actual_negatives개만 선택
        _, top_indices = tf.nn.top_k(-tf.cast(random_indices, tf.float32), k=actual_negatives)
        
        # 필요한 개수만큼 확장 (self.num_negatives에 맞춤)
        if actual_negatives < self.num_negatives:
            # 반복하여 필요한 크기까지 확장
            repeat_times = tf.cast(tf.math.ceil(self.num_negatives / actual_negatives), tf.int32)
            repeated_indices = tf.tile(top_indices, [1, repeat_times])
            # 정확한 크기로 자르기
            neg_indices = repeated_indices[:, :self.num_negatives]
        else:
            neg_indices = top_indices[:, :self.num_negatives]
        
        # [batch_size, sequence_length, num_negatives] 형태로 확장
        neg_indices = tf.tile(tf.expand_dims(neg_indices, axis=1), [1, sequence_length, 1])
        
        return neg_indices 


# wav2vec2 음성 인식(ASR) 모델 구현
class Wav2Vec2ForCTC(tf.keras.Model):
    def __init__(self, config):
        super(Wav2Vec2ForCTC, self).__init__()
        self.config = config
        
        # 기본 wav2vec2 모델
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # CTC를 위한 투영 레이어
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.lm_head = tf.keras.layers.Dense(config.vocab_size, name="lm_head")
        
        # CTC 손실 설정
        self.ctc_loss_reduction = config.ctc_loss_reduction
        self.ctc_zero_infinity = config.ctc_zero_infinity
    
    def call(self, inputs, attention_mask=None, labels=None, 
             output_attentions=False, output_hidden_states=False,
             return_dict=True, training=False):
        
        # wav2vec2 모델 호출
        outputs = self.wav2vec2(
            inputs,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training
        )
        
        # 최종 히든 스테이트
        hidden_states = outputs["last_hidden_state"]
        hidden_states = self.dropout(hidden_states, training=training)
        
        # 로짓 계산
        logits = self.lm_head(hidden_states)
        
        # 손실 계산 (학습 중이고 레이블이 제공된 경우)
        loss = None
        if training and labels is not None:
            # CTC 손실 계산
            loss = self._compute_ctc_loss(logits, labels, attention_mask)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states", None),
            "attentions": outputs.get("attentions", None)
        }
    
    def _compute_ctc_loss(self, logits, labels, attention_mask):
        """CTC 손실 계산"""
        # CTC 손실 구현 (실제 구현에서는 더 복잡한 로직 필요)
        # 여기서는 간단한 더미 손실 반환
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.zeros_like(logits[:,:,0], dtype=tf.int32), 
            logits=logits
        ))


# 음성 분류 모델 구현 (예: 감정 분류)
class Wav2Vec2ForSequenceClassification(tf.keras.Model):
    def __init__(self, config):
        super(Wav2Vec2ForSequenceClassification, self).__init__()
        self.config = config
        
        # 기본 wav2vec2 모델
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # 분류를 위한 추가 레이어
        self.projector = tf.keras.layers.Dense(config.classifier_proj_size, activation="tanh", name="classifier_proj")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout)
        self.classifier = tf.keras.layers.Dense(config.num_labels, name="classifier")
    
    def call(self, inputs, attention_mask=None, labels=None,
             output_attentions=False, output_hidden_states=False,
             return_dict=True, training=False):
        
        # wav2vec2 모델 호출
        outputs = self.wav2vec2(
            inputs,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training
        )
        
        # 최종 히든 스테이트에서 시퀀스 임베딩 추출 (평균 풀링)
        hidden_states = outputs["last_hidden_state"]
        
        # 마스크 적용 (있는 경우)
        if attention_mask is not None:
            # 마스크 확장하여 히든 스테이트와 동일한 차원으로 만들기
            float_mask = tf.cast(tf.expand_dims(attention_mask, -1), tf.float32)
            # 마스크된 평균 계산 (마스크가 0인 부분은 제외)
            pooled_output = tf.reduce_sum(hidden_states * float_mask, axis=1) / tf.reduce_sum(float_mask, axis=1)
        else:
            # 마스크가 없는 경우 단순 평균 계산
            pooled_output = tf.reduce_mean(hidden_states, axis=1)
        
        # 분류 레이어 적용
        pooled_output = self.projector(pooled_output)
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        
        # 손실 계산 (학습 중이고 레이블이 제공된 경우)
        loss = None
        if training and labels is not None:
            # 분류 손실 계산 (예: 교차 엔트로피)
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                y_true=labels, y_pred=logits, from_logits=True
            )
            loss = tf.reduce_mean(loss)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.get("hidden_states", None),
            "attentions": outputs.get("attentions", None)
        } 


# ============ 유틸리티 함수 및 학습 관련 코드 ============ #

# 시간 마스킹 적용 함수
def apply_time_mask(hidden_states, mask_prob=0.05, mask_length=10):
    """오디오 시퀀스의 시간 차원에 마스킹 적용"""
    batch_size, sequence_length, hidden_size = tf.shape(hidden_states)[0], tf.shape(hidden_states)[1], tf.shape(hidden_states)[2]
    
    # 마스크 생성
    mask = tf.random.uniform(shape=(batch_size, sequence_length)) < mask_prob
    
    # 연속적인 마스킹 확장 (mask_length만큼 연속으로 마스킹)
    expanded_mask = tf.zeros_like(mask, dtype=tf.bool)
    
    for i in range(mask_length):
        # 마스크를 i 위치만큼 이동하여 원래 마스크와 OR 연산
        shifted_mask = tf.pad(mask[:, :sequence_length-i], [[0, 0], [i, 0]], constant_values=False)
        expanded_mask = tf.logical_or(expanded_mask, shifted_mask)
    
    # 확장된 마스크를 히든 스테이트와 동일한 차원으로 만들기
    expanded_mask = tf.cast(tf.expand_dims(expanded_mask, -1), tf.float32)
    
    # 마스킹된 위치는 0으로 설정
    masked_hidden_states = hidden_states * (1.0 - expanded_mask)
    
    return masked_hidden_states, expanded_mask


# 특징 마스킹 적용 함수
def apply_feature_mask(hidden_states, mask_prob=0.05, mask_length=10):
    """오디오 특징 차원에 마스킹 적용"""
    batch_size, sequence_length, hidden_size = tf.shape(hidden_states)[0], tf.shape(hidden_states)[1], tf.shape(hidden_states)[2]
    
    # 차원별 마스크 생성
    mask = tf.random.uniform(shape=(batch_size, hidden_size)) < mask_prob
    
    # 연속적인 마스킹 확장 (mask_length만큼 연속으로 마스킹)
    expanded_mask = tf.zeros_like(mask, dtype=tf.bool)
    
    for i in range(mask_length):
        # 마스크를 i 위치만큼 이동하여 원래 마스크와 OR 연산
        shifted_mask = tf.pad(mask[:, :hidden_size-i], [[0, 0], [i, 0]], constant_values=False)
        expanded_mask = tf.logical_or(expanded_mask, shifted_mask)
    
    # 확장된 마스크를 히든 스테이트와 동일한 차원으로 만들기
    expanded_mask = tf.cast(tf.expand_dims(expanded_mask, 1), tf.float32)
    
    # 마스킹된 위치는 0으로 설정
    masked_hidden_states = hidden_states * (1.0 - expanded_mask)
    
    return masked_hidden_states, expanded_mask


# 더미 오디오 데이터셋 생성 (수정됨)
def create_dummy_dataset(batch_size):
    """
    형태가 일관된 더미 오디오 데이터셋 생성 - 메모리 절약을 위해 길이 단축
    """
    def generate_consistent_data():
        # 메모리 절약을 위해 더 짧은 오디오 데이터 생성 (2초)
        audio_length = 32000  # 2초 * 16kHz (기존 5초에서 단축)
        
        # 더미 오디오 특징 생성
        audio_features = tf.random.normal([audio_length], dtype=tf.float32)
        
        # 더미 레이블
        label = tf.constant(0.0, dtype=tf.float32)
        
        return audio_features, label
    
    # 데이터셋 생성
    dataset = tf.data.Dataset.from_generator(
        lambda: [generate_consistent_data() for _ in range(50)],
        output_signature=(
            tf.TensorSpec(shape=[32000], dtype=tf.float32),  # 길이 변경
            tf.TensorSpec(shape=[], dtype=tf.float32)
        )
    )
    
    # 배치 처리 - drop_remainder=True로 일관된 배치 크기 보장
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset


# 전체 모델과 데이터셋을 연결하여 단일 GPU 학습 실행
def create_full_model(model_type="pretraining", 
                      model_size="small",  # 추가: 모델 크기 선택
                      num_negatives=100,
                      mask_time_prob=0.065,
                      mask_time_length=10):
    """전체 wav2vec2 모델 생성"""
    
    # 모델 설정
    config = Wav2Vec2Config(model_size=model_size)
    
    # 필요한 모델 유형에 맞게 설정 수정
    config.num_negatives = num_negatives
    config.mask_time_prob = mask_time_prob
    config.mask_time_length = mask_time_length
    
    if model_type == "pretraining":
        return Wav2Vec2ForPreTraining(config)
    elif model_type == "asr":
        # 음성인식 (ASR)을 위한 모델 반환
        return Wav2Vec2ForCTC(config)
    elif model_type == "classification":
        # 분류를 위한 모델 반환
        return Wav2Vec2ForSequenceClassification(config)
    else:
        # 기본 모델 반환
        return Wav2Vec2Model(config)


# 단일 GPU 학습 스텝 정의 (수정됨)
@tf.function
def train_step(model, inputs, optimizer):
    """개선된 단일 GPU 학습 스텝"""
    features, labels = inputs
    
    # 배치 크기 검증
    batch_size = tf.shape(features)[0]
    
    # 빈 배치인 경우 0 손실 반환
    if tf.equal(batch_size, 0):
        return tf.constant(0.0, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        # 모델 순전파
        outputs = model(features, training=True)
        
        # 손실 계산
        if isinstance(model, Wav2Vec2ForPreTraining):
            if "projected_states" in outputs and "projected_quantized_features" in outputs:
                # 컨트라스트 손실 계산
                logits, contrastive_loss = model._compute_contrastive_loss(
                    outputs["projected_states"],
                    outputs["projected_quantized_features"]
                )
                
                # 다양성 손실 추가
                if "codevector_perplexity" in outputs:
                    diversity_loss = model._compute_diversity_loss(outputs["codevector_perplexity"])
                else:
                    diversity_loss = tf.constant(0.0)
                
                # 최종 손실
                loss = contrastive_loss + model.diversity_loss_weight * diversity_loss
            else:
                loss = tf.constant(0.0)
        else:
            outputs = model(features, labels=labels, training=True)
            loss = outputs.get("loss", 0.0)
        
        # NaN 체크
        loss = tf.where(tf.math.is_nan(loss), tf.constant(0.0), loss)
    
    # 그래디언트 계산
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # None 그래디언트 필터링
    gradients = [
        tf.zeros_like(var) if grad is None else grad 
        for grad, var in zip(gradients, model.trainable_variables)
    ]
    
    # 그래디언트 클리핑
    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
    
    # 옵티마이저 적용
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return loss


# wav2vec2 모델 메인 학습 함수 (단일 GPU 버전)
def train_wav2vec2(model_type="pretraining", model_size="small", num_epochs=1, learning_rate=3e-5, batch_size=8):
    """단일 GPU wav2vec2 모델 학습 함수"""
    
    # 모델 생성
    model = create_full_model(model_type=model_type, model_size=model_size)
    
    # 데이터셋 생성 (단일 GPU용) - 모델 초기화를 위해 먼저 생성
    train_dataset = create_dummy_dataset(batch_size)
    iterator = iter(train_dataset)
    
    # 모델 초기화를 위해 첫 번째 배치로 한 번 호출
    try:
        first_batch = next(iterator)
        print("모델 가중치 초기화 중...")
        # 모델 호출로 가중치 생성
        _ = model(first_batch[0], training=False)
        print("모델 가중치 초기화 완료")
    except Exception as e:
        print(f"모델 초기화 중 오류 발생: {e}")
        # 다시 iterator 생성
        iterator = iter(train_dataset)
    
    # 옵티마이저 설정
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        epsilon=1e-8,
        clipnorm=1.0
    )
    
    # 메트릭 설정
    metrics = []
    if model_type == "pretraining":
        metrics.extend([
            tf.keras.metrics.Mean(name="contrastive_loss"),
            tf.keras.metrics.Mean(name="diversity_loss"),
            tf.keras.metrics.Mean(name="perplexity")
        ])
    elif model_type == "asr":
        metrics.append(tf.keras.metrics.Mean(name="ctc_loss"))
    elif model_type == "classification":
        metrics.extend([
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Mean(name="loss")
        ])
    
    # 모델 컴파일
    model.compile(optimizer=optimizer, metrics=metrics)
    
    # 체크포인트 설정
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    
    # 학습 루프
    step = 0
    
    # 시작 시간 기록
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"에포크 {epoch+1}/{num_epochs}")
        
        for _ in range(MAX_ITERATIONS):
            # 데이터셋에서 배치 가져오기
            try:
                inputs = next(iterator)
            except StopIteration:
                iterator = iter(train_dataset)
                inputs = next(iterator)
            
            # 현재 시간 기록
            step_start = time.time()
            
            # 단일 GPU 학습 스텝 실행
            loss = train_step(model, inputs, optimizer)
            
            # 스텝 완료 시간
            step_end = time.time()
            step_duration = step_end - step_start
            elapsed = step_end - start_time
            
            # 모든 스텝에 대해 타임스탬프와 함께 로깅
            try:
                loss_value = float(loss)
            except:
                loss_value = 0.0
            print(f"Step {step}, Loss: {loss_value:.4f}, Time: {time.strftime('%H:%M:%S')} (경과: {elapsed:.2f}초, 스텝 시간: {step_duration:.2f}초)")
            
            step += 1
            
            # 주기적 체크포인트 저장
            if step % 50 == 0:
                checkpoint.save(os.path.join(checkpoint_dir, f"model_step_{step}"))
        
        # 에포크 종료 후 체크포인트 저장
        checkpoint.save(os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}"))
    
    return model


# 메인 함수
def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Wav2Vec2 Single GPU Speech Recognition')
    parser.add_argument('--num_batches', type=int, default=5, help='num_batches, default is set 5')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size, default is set 1')
    parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'small', 'base'], 
                       help='Model size: tiny (~15-20M params), small (~30-40M params), base (~95M params)')
    parser.add_argument('--model_type', type=str, default='pretraining', choices=['pretraining', 'asr', 'classification'],
                       help='Model type for training')
    parser.add_argument('--learning_rate', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
    args = parser.parse_args()

    # 전역 변수 설정
    global MAX_ITERATIONS
    MAX_ITERATIONS = args.num_batches
    
    # 모델과 프로세서를 저장할 로컬 디렉토리 설정
    CACHE_DIR = './model_cache'
    os.makedirs(CACHE_DIR, exist_ok=True)

    print("Wav2Vec2 단일 GPU 학습 시작...")
    print(f"선택된 모델 크기: {args.model_size}")
    print(f"선택된 모델 타입: {args.model_type}")
    
    # 모델 크기별 파라미터 수 안내
    if args.model_size == "tiny":
        print("Tiny 모델: 약 15-20M 파라미터")
    elif args.model_size == "small":
        print("Small 모델: 약 30-40M 파라미터")
    else:
        print("Base 모델: 약 95M 파라미터")
    
    # 메모리 최적화 설정
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU 메모리 설정 오류: {e}")
    
    # JCT 측정 시작
    start_time = time.time()
    
    # 모델 학습 실행
    model = train_wav2vec2(
        model_type=args.model_type, 
        model_size=args.model_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size
    )
    
    # JCT 측정 종료
    end_time = time.time()
    jct = end_time - start_time
    
    # 결과 출력
    print("학습 완료.")
    print("JCT:", jct)
    
    # 모델 저장
    model_path = os.path.join(CACHE_DIR, f"wav2vec2_{args.model_size}_{args.model_type}_model")
    model.save_weights(model_path)
    print(f"{args.model_size.capitalize()} {args.model_type} 모델이 {model_path}에 저장되었습니다.")


if __name__ == "__main__":
    main() 