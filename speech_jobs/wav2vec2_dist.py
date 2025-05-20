import tensorflow as tf
import numpy as np
import json
import os
import sys
import time
import argparse


class Wav2Vec2Config:
    def __init__(self):
        # 특징 추출기 설정
        self.feat_extract_norm = "group"
        self.feat_extract_activation = "gelu"
        self.conv_dim = [512, 512, 512, 512, 512, 512, 512]
        self.conv_stride = [5, 2, 2, 2, 2, 2, 2]
        self.conv_kernel = [10, 3, 3, 3, 3, 2, 2]
        self.conv_bias = False
        self.num_conv_pos_embeddings = 128
        self.num_conv_pos_embedding_groups = 16
        
        # 모델 차원 및 인코더 설정
        self.hidden_size = 768
        self.num_hidden_layers = 12
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.hidden_act = "gelu"
        self.hidden_dropout = 0.1
        self.activation_dropout = 0.1
        self.attention_dropout = 0.1
        self.layer_norm_eps = 1e-5
        
        # 양자화 및 마스킹 설정
        self.num_codevectors_per_group = 320
        self.num_codevector_groups = 2
        self.contrastive_logits_temperature = 0.1
        self.num_negatives = 100
        self.codevector_dim = 256
        self.proj_codevector_dim = 256
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
        self.classifier_proj_size = 256
        self.tdnn_dim = [512, 512, 512, 512, 1500]
        self.tdnn_kernel = [5, 3, 3, 1, 1]
        self.tdnn_dilation = [1, 2, 3, 1, 1]
        self.xvector_output_dim = 512


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
            filters=config.hidden_size,
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

# Wav2Vec2Quantizer 클래스 구현
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
        
        # 이 벡터는 contrastive loss에서 사용될 negative samples 관리에 필요
        self.neg_sample_indices = None
    
    def call(self, hidden_states, training=False):
        batch_size, sequence_length, hidden_size = tf.shape(hidden_states)[0], tf.shape(hidden_states)[1], tf.shape(hidden_states)[2]
        
        # hidden_states를 그룹 수에 맞게 재구성
        hidden_states = tf.reshape(
            hidden_states, 
            [batch_size, sequence_length, self.config.num_codevector_groups, hidden_size // self.config.num_codevector_groups]
        )
        
        # 각 그룹별로 코드벡터와의 거리 계산
        distances = []
        for i in range(self.config.num_codevector_groups):
            group_hidden = hidden_states[:, :, i, :]  # [batch, time, dim]
            
            # hidden_states를 [batch*time, dim] 형태로 변환
            reshaped_hidden = tf.reshape(group_hidden, [-1, tf.shape(group_hidden)[-1]])  # [batch*time, dim]
            
            # 현재 그룹의 코드벡터들 [num_vectors, dim]
            group_vectors = self.codevectors[i]  # [num_vectors, dim]
            
            # 각 hidden state와 모든 코드벡터 사이의 유클리드 거리 계산
            expanded_hidden = tf.expand_dims(reshaped_hidden, 1)  # [batch*time, 1, dim]
            expanded_vectors = tf.expand_dims(group_vectors, 0)   # [1, num_vectors, dim]
            
            # 거리 계산: [batch*time, num_vectors]
            dist = tf.reduce_sum(tf.square(expanded_hidden - expanded_vectors), axis=-1)
            
            # 원래 배치, 시간 차원으로 복원: [batch, time, num_vectors]
            dist = tf.reshape(dist, [batch_size, sequence_length, -1])
            distances.append(dist)
        
        # 모든 그룹의 거리: [num_groups, batch, time, num_vectors]
        distances = tf.stack(distances, axis=0)
        
        # 가장 가까운 코드벡터 선택 (Hard Quantization)
        indices = tf.argmin(distances, axis=-1)  # [num_groups, batch, time]
        
        # one-hot 인코딩으로 변환
        encodings = tf.one_hot(indices, self.config.num_codevectors_per_group)  # [num_groups, batch, time, num_vectors]
        
        # 가장 가까운 코드벡터 가져오기
        quantized_features = []
        for i in range(self.config.num_codevector_groups):
            # 인코딩을 사용하여 코드벡터 선택
            encoding = encodings[i]  # [batch, time, num_vectors]
            
            # 코드벡터와 내적 계산
            # [batch, time, num_vectors] @ [num_vectors, dim] -> [batch, time, dim]
            quantized = tf.matmul(encoding, self.codevectors[i])
            quantized_features.append(quantized)
        
        # 그룹별 양자화 결과 결합
        # [batch, time, num_groups, dim]
        quantized_features = tf.stack(quantized_features, axis=2)
        
        # 원래 모양으로 재구성
        # [batch, time, hidden_size]
        quantized_features = tf.reshape(quantized_features, [batch_size, sequence_length, -1])
        
        # 다양성 계산 (코드북 활용도)
        avg_probs = tf.reduce_mean(encodings, axis=[1, 2])  # [num_groups, num_vectors]
        perplexity = tf.exp(-tf.reduce_sum(avg_probs * tf.math.log(avg_probs + 1e-10), axis=-1))  # [num_groups]
        perplexity = tf.reduce_mean(perplexity)  # 스칼라
        
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
        # 실제 구현은 생략되었지만 실제 사용 시 구현 필요
        self.quantizer = Wav2Vec2Quantizer(config)

        # 투영 헤드
        self.project_hid = Wav2Vec2ProjectionHead(config)
        self.project_q = Wav2Vec2ProjectionHead(config)
    
    def call(self, inputs, attention_mask=None, 
             output_attentions=False, output_hidden_states=False,
             return_dict=True, training=False):
        
        # 1. 특징 추출
        extract_features = self.feature_extractor(inputs, training=training)
        
        # 2. 타겟 양자화를 위한 특징 제공
        if training:
            # 양자화 대상 특징
            quantize_targets = extract_features
            
            # 양자화
            quantized_result = self.quantizer(quantize_targets, training=True)
            quantized_features = quantized_result["quantized_features"]
            codevector_perplexity = quantized_result["codevector_perplexity"]
        else:
            quantized_features = None
            codevector_perplexity = None
        
        # 3. 특징 투영
        hidden_states = self.feature_projection(extract_features)
        hidden_states = self.feature_projection_layer_norm(hidden_states)
        hidden_states = self.feature_projection_dropout(hidden_states, training=training)
        
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

# 사전학습 wav2vec2 모델 구현
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
        if training:
            # 인코더 출력 프로젝션
            transformer_features = self.wav2vec2.project_hid(outputs["last_hidden_state"])
            
            # 양자화된 특징 프로젝션
            quantized_features = self.wav2vec2.project_q(outputs["quantized_features"])
            
            # 손실 계산을 위한 정보 추가
            outputs["projected_quantized_features"] = quantized_features
            outputs["projected_states"] = transformer_features
        
        return outputs
    
    def train_step(self, data):
        inputs, _ = data
        
        with tf.GradientTape() as tape:
            # 모델 순전파
            outputs = self(inputs, training=True)
            
            # 대조 손실 계산
            logits, loss = self._compute_contrastive_loss(
                outputs["projected_states"],
                outputs["projected_quantized_features"]
            )
            
            # 다양성 손실 추가
            diversity_loss = self._compute_diversity_loss(outputs["codevector_perplexity"])
            
            # 최종 손실
            total_loss = loss + self.diversity_loss_weight * diversity_loss
        
        # 그래디언트 계산 및 적용
        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 손실 기록
        self.compiled_metrics.update_state(0, total_loss)
        
        results = {m.name: m.result() for m in self.metrics}
        results.update({
            "contrastive_loss": loss,
            "diversity_loss": diversity_loss,
            "perplexity": outputs["codevector_perplexity"]
        })
        
        return results
    
    def _compute_contrastive_loss(self, hidden_states, quantized_states):
        """컨트라스트 손실 계산"""
        batch_size, sequence_length, hidden_size = tf.shape(hidden_states)[0], tf.shape(hidden_states)[1], tf.shape(hidden_states)[2]
        
        # 양수 쌍 (같은 시간 위치의 인코더 출력과 양자화된 특징)
        pos_logits = tf.reduce_sum(hidden_states * quantized_states, axis=-1) / self.contrastive_logits_temperature
        
        # 음수 쌍
        if self.num_negatives > 0:
            # 타임프레임을 섞어 부정 샘플 생성
            neg_indices = self._sample_negative_indices(sequence_length, batch_size)
            
            # 부정 샘플 가져오기
            neg_quantized = tf.gather(quantized_states, neg_indices, axis=1, batch_dims=1)
            
            # hidden_states: [batch, time, dim], neg_quantized: [batch, time, neg, dim]
            # 확장하여 효율적인 내적 계산
            hidden_states_expanded = tf.expand_dims(hidden_states, axis=2)  # [batch, time, 1, dim]
            
            # 음수 로짓 계산
            neg_logits = tf.reduce_sum(hidden_states_expanded * neg_quantized, axis=-1) / self.contrastive_logits_temperature
            
            # 양수와 음수 로짓 결합
            logits = tf.concat([tf.expand_dims(pos_logits, axis=2), neg_logits], axis=2)
        else:
            logits = tf.expand_dims(pos_logits, axis=2)
        
        # 첫 번째 인덱스 (양수 쌍)에 대한 예측 손실 계산
        labels = tf.zeros(tf.shape(logits)[:-1], dtype=tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        
        # 마스크가 있는 경우 손실에 적용
        # (여기서는 간소화를 위해 모든 위치가 유효하다고 가정)
        
        return logits, tf.reduce_mean(loss)
    
    def _compute_diversity_loss(self, perplexity):
        """다양성 손실 계산 - 코드북 활용도를 높이기 위함"""
        # 퍼플렉시티가 높을수록 다양성이 높음을 의미
        # 인위적으로 다양성을 높이기 위해 -perplexity를 최소화
        return -perplexity
    
    def _sample_negative_indices(self, sequence_length, batch_size):
        """컨트라스트 학습을 위한 부정 샘플 인덱스 생성"""
        # [0, 1, 2, ..., sequence_length - 1]
        all_indices = tf.range(sequence_length)
        
        # 배치별로 다른 무작위 인덱스 생성
        neg_indices = tf.stack([
            tf.random.shuffle(all_indices) for _ in range(batch_size)
        ], axis=0)
        
        # 각 위치에 대해 num_negatives 개의 음수 샘플 선택
        # [batch, time, num_negatives]
        neg_indices = tf.stack([
            tf.roll(neg_indices, shift=i+1, axis=1)[:, :self.num_negatives] 
            for i in range(sequence_length)
        ], axis=1)
        
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
    
    def _compute_ctc_loss(self, logits, labels, attention_mask=None):
        """CTC 손실 계산"""
        # 로짓 길이 (time steps) 계산
        input_lengths = tf.ones(tf.shape(logits)[0], dtype=tf.int32) * tf.shape(logits)[1]
        
        # 마스크가 있는 경우 길이 조정
        if attention_mask is not None:
            input_lengths = tf.reduce_sum(attention_mask, axis=1, keepdims=False)
            
        # 레이블 길이 계산
        label_lengths = tf.reduce_sum(tf.cast(labels > 0, tf.int32), axis=1, keepdims=False)
        
        # CTC 손실 계산
        loss = tf.nn.ctc_loss(
            labels=tf.cast(labels, tf.int32),
            logits=tf.transpose(logits, perm=[1, 0, 2]),  # CTC는 time-major 형식 필요
            label_length=label_lengths,
            logit_length=input_lengths,
            blank_index=0,
            logits_time_major=True
        )
        
        # 무한 손실 처리
        if self.ctc_zero_infinity:
            loss = tf.where(tf.math.is_inf(loss), tf.zeros_like(loss), loss)
        
        # 손실 축소 방법
        if self.ctc_loss_reduction == "mean":
            loss = tf.reduce_mean(loss)
        elif self.ctc_loss_reduction == "sum":
            loss = tf.reduce_sum(loss)
        
        return loss
    
    def train_step(self, data):
        inputs, labels = data
        
        with tf.GradientTape() as tape:
            # 모델 순전파
            outputs = self(inputs, labels=labels, training=True)
            loss = outputs["loss"]
        
        # 그래디언트 계산 및 적용
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 손실 기록
        self.compiled_metrics.update_state(0, loss)
        
        # 결과 반환
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        
        return results


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
    
    def train_step(self, data):
        inputs, labels = data
        
        with tf.GradientTape() as tape:
            # 모델 순전파
            outputs = self(inputs, labels=labels, training=True)
            loss = outputs["loss"]
        
        # 그래디언트 계산 및 적용
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # 손실 기록
        self.compiled_metrics.update_state(labels, outputs["logits"])
        
        # 결과 반환
        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": loss})
        
        return results


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


# 전체 모델과 데이터셋을 연결하여 분산 학습 실행
def create_full_model(model_type="pretraining", 
                      num_negatives=100,
                      mask_time_prob=0.065,
                      mask_time_length=10):
    """전체 wav2vec2 모델 생성"""
    
    # 모델 설정
    config = Wav2Vec2Config()
    
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


# 분산 학습 스텝 정의
@tf.function
def distributed_train_step(strategy, model, dist_inputs, optimizer):
    """분산 학습을 위한 스텝 함수"""
    
    def train_step(inputs):
        features, labels = inputs
        
        with tf.GradientTape() as tape:
            # 모델 타입에 따라 다른 호출 방식 적용
            # Wav2Vec2ForPreTraining은 labels 매개변수를 받지 않음
            if isinstance(model, Wav2Vec2ForPreTraining):
                outputs = model(features, training=True)
                
                # 컨트라스트 손실 계산
                logits, contrastive_loss = model._compute_contrastive_loss(
                    outputs["projected_states"],
                    outputs["projected_quantized_features"]
                )
                
                # 다양성 손실 추가
                diversity_loss = model._compute_diversity_loss(outputs["codevector_perplexity"])
                
                # 최종 손실
                loss = contrastive_loss + model.diversity_loss_weight * diversity_loss
            else:
                # ASR이나 분류 모델의 경우 labels 매개변수 전달
                outputs = model(features, labels=labels, training=True)
                loss = outputs["loss"]
        
        # 그래디언트 계산 및 적용
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    # 분산 전략으로 학습 스텝 실행
    per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
    
    # 손실 값 집계
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# wav2vec2 모델 메인 학습 함수
def train_wav2vec2(strategy, model_type="pretraining", num_epochs=5, learning_rate=3e-5):
    """wav2vec2 모델 학습 함수"""
    with strategy.scope():
        # 모델 생성
        model = create_full_model(model_type=model_type)
        
        # 옵티마이저 설정
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
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
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for _ in range(MAX_ITERATIONS):
            # 분산 데이터셋에서 배치 가져오기
            try:
                inputs = next(iterator)
            except StopIteration:
                iterator = iter(dist_dataset)
                inputs = next(iterator)
            
            # 분산 학습 스텝 실행 (strategy 명시적 전달)
            loss = distributed_train_step(strategy, model, inputs, optimizer)
            
            # 로깅
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.numpy():.4f}")
            
            step += 1
        
        # 에포크 종료 후 체크포인트 저장
        checkpoint.save(os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}"))
    
    return model


# 메인 함수
def main(strategy):
    print("Wav2Vec2 분산 학습 시작...")
    
    # 네트워크 및 GPU 모니터링 시작
    os.system('sh /workspace/network.sh &')  # network profile
    os.system('sh /workspace/gpu.sh &')  # gpu profile
    print('''
========================
network profile started!
========================''')
    
    # JCT 측정 시작
    start_time = time.time()
    start_time_tf = tf.timestamp()
    
    # 모델 학습 실행
    model = train_wav2vec2(strategy, model_type="pretraining")
    
    # JCT 측정 종료
    end_time = time.time()
    jct = end_time - start_time
    
    # 결과 출력
    print("Training completed.")
    print("jct:", jct)
    
    # JCT 파일 저장
    model_txt = open('/workspace/model.txt', 'r')
    save_dir_name = model_txt.read()
    jct_file = open('/result/' + save_dir_name.strip() + '/' + task_type + '_' + str(task_index) + '_jct.txt', 'w')
    jct_file.write('%.2f' % (float(jct)))
    jct_file.close()
    model_txt.close()
    
    # 모델 저장
    model_path = os.path.join(CACHE_DIR, "wav2vec2_model")
    model.save_weights(model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")


if __name__ == "__main__":
    # 명령줄 인자 파싱
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
    
    main(strategy)