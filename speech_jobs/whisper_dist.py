import tensorflow as tf
import numpy as np
import json
import os
import time
import argparse
from tensorflow.keras import layers, Model


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
class PositionalEncoding(tf.keras.layers.Layer):
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
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


# 멀티헤드 어텐션 구현
class MultiHeadAttention(tf.keras.layers.Layer):
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
        batch_size = tf.shape(hidden_states)[0]
        seq_len = tf.shape(hidden_states)[1]
        
        # cross-attention인 경우 key, value는 인코더 출력, query는 디코더 상태
        is_cross_attention = key_value_states is not None
        
        if is_cross_attention:
            # cross-attention인 경우 key_value_states에서 key와 value 추출
            key_states = self._reshape(self.k_proj(key_value_states))  # [batch, num_heads, kv_seq_len, head_dim]
            value_states = self._reshape(self.v_proj(key_value_states))  # [batch, num_heads, kv_seq_len, head_dim]
            kv_seq_len = tf.shape(key_states)[2]
        elif past_key_value is not None:
            # 과거 키/값이 있는 경우 (디코더의 auto-regressive 생성 시)
            key_states = self._reshape(self.k_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
            value_states = self._reshape(self.v_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
            
            # 과거 키/값과 현재 키/값 연결
            key_states = tf.concat([past_key_value[0], key_states], axis=2)
            value_states = tf.concat([past_key_value[1], value_states], axis=2)
            kv_seq_len = tf.shape(key_states)[2]
        else:
            # 일반적인 self-attention
            key_states = self._reshape(self.k_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
            value_states = self._reshape(self.v_proj(hidden_states))  # [batch, num_heads, seq_len, head_dim]
            kv_seq_len = seq_len
        
        # 항상 쿼리는 현재 hidden_states에서 계산
        query_states = self._reshape(self.q_proj(hidden_states) * self.scaling)  # [batch, num_heads, seq_len, head_dim]
        
        # 현재 키/값 저장 (디코더에서 캐싱 시 사용)
        past_key_value = (key_states, value_states) if self.is_decoder else None
        
        # 어텐션 스코어 계산: [batch, num_heads, seq_len, kv_seq_len]
        attention_scores = tf.matmul(query_states, key_states, transpose_b=True)
        
        # 어텐션 마스크 적용 (존재하는 경우)
        if attention_mask is not None:
            # 마스크 확장 및 적용 (마스크가 0인 위치는 -inf로 설정)
            attention_mask = tf.cast(attention_mask, tf.float32)
            attention_mask = (1.0 - attention_mask) * -1e9
            attention_scores = attention_scores + attention_mask
        
        # 소프트맥스 적용
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        
        # 드롭아웃 적용
        attention_probs = self.dropout(attention_probs, training=training)
        
        # 헤드 마스크 적용 (필요한 경우)
        if layer_head_mask is not None:
            attention_probs = attention_probs * tf.expand_dims(tf.expand_dims(layer_head_mask, -1), -1)
        
        # 어텐션 출력 계산
        attention_output = tf.matmul(attention_probs, value_states)  # [batch, num_heads, seq_len, head_dim]
        
        # 출력 형태 변환
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])  # [batch, seq_len, num_heads, head_dim]
        attention_output = tf.reshape(attention_output, (batch_size, seq_len, self.d_model))  # [batch, seq_len, d_model]
        
        # 최종 선형 변환
        attention_output = self.out_proj(attention_output)
        
        return attention_output, attention_probs, past_key_value


# 피드포워드 네트워크
class FeedForward(tf.keras.layers.Layer):
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
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.activation_dropout(hidden_states, training=training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
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
        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        attention_output, attention_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            training=training
        )
        hidden_states = residual + attention_output
        
        # Feed Forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states, training=training)
        hidden_states = residual + feed_forward_output
        
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
        
        # 캐시된 과거 키/값 분리
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        cross_attn_past_key_value = past_key_value[2:] if past_key_value is not None else None
        
        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        
        attention_output, self_attention_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            training=training
        )
        
        hidden_states = residual + attention_output
        
        # Cross Attention
        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        
        cross_attention_output, cross_attention_weights, cross_attn_present_key_value = self.encoder_attn(
            hidden_states=hidden_states,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            training=training
        )
        
        hidden_states = residual + cross_attention_output
        
        # Feed Forward
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        feed_forward_output = self.feed_forward(hidden_states, training=training)
        hidden_states = residual + feed_forward_output
        
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
        # 차원 변환 (채널 마지막)
        input_features = tf.transpose(input_features, perm=[0, 2, 1])  # [batch, seq_len, n_mels]
        
        # 컨볼루션 레이어 적용
        hidden_states = self.conv1(input_features)
        hidden_states = tf.keras.activations.gelu(hidden_states)
        
        hidden_states = self.conv2(hidden_states)
        hidden_states = tf.keras.activations.gelu(hidden_states)
        
        # 위치 인코딩 추가
        hidden_states = self.positional_embedding(hidden_states)
        
        # 드롭아웃
        hidden_states = self.dropout(hidden_states, training=training)
        
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
        
        # 최종 레이어 정규화
        hidden_states = self.layer_norm(hidden_states)
        
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
        batch_size, seq_length = tf.shape(input_ids)[0], tf.shape(input_ids)[1]
        
        # 입력 토큰 임베딩
        inputs_embeds = self.embed_tokens(input_ids)
        
        # 위치 인코딩 추가
        hidden_states = self.positional_embedding(inputs_embeds)
        
        # 드롭아웃
        hidden_states = self.dropout(hidden_states, training=training)
        
        # 어텐션 마스크 확인 및 생성
        if attention_mask is None:
            # 인과적 마스크 생성 (자기 자신과 이전 위치만 볼 수 있음)
            attention_mask = 1.0 - tf.linalg.band_part(
                tf.ones((seq_length, seq_length)), -1, 0)
            attention_mask = tf.expand_dims(attention_mask, 0)  # [1, seq_len, seq_len]
        
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
            
            if use_cache:
                present_key_values = present_key_values + (present_key_value,)
        
        # 최종 레이어 정규화
        hidden_states = self.layer_norm(hidden_states)
        
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
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_features,
                attention_mask=attention_mask,
                training=training
            )
        
        # 인코더 출력
        encoder_hidden_states = encoder_outputs["last_hidden_state"]
        
        # 디코더 입력이 제공되지 않은 경우 시작 토큰 생성
        if decoder_input_ids is None:
            batch_size = tf.shape(input_features)[0]
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
        
        # 손실 계산 (학습 중이고 레이블이 제공된 경우)
        loss = None
        if training and labels is not None:
            # 손실 계산을 위해 레이블 시프트 (teacher forcing)
            shift_labels = labels[:, 1:]
            shift_logits = lm_logits[:, :-1, :]
            
            # 손실 계산 (교차 엔트로피)
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.keras.losses.Reduction.NONE
            )
            
            loss = loss_fn(shift_labels, shift_logits)
            
            # 패딩 토큰 마스킹 (패딩 토큰은 손실 계산에서 제외)
            if decoder_attention_mask is not None:
                loss = loss * tf.cast(decoder_attention_mask[:, :-1], dtype=loss.dtype)
                loss = tf.reduce_sum(loss) / tf.reduce_sum(tf.cast(decoder_attention_mask[:, :-1], dtype=loss.dtype))
            else:
                loss = tf.reduce_mean(loss)
        
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
    
    # 더미 레이블 (토큰 ID) 생성
    dummy_labels = np.random.randint(
        low=3,  # 특수 토큰 ID 이후부터 시작
        high=100,  # 소규모 가상 어휘
        size=(num_samples, max_target_length)
    ).astype(np.int32)
    
    # TensorFlow 데이터셋 생성
    dataset = tf.data.Dataset.from_tensor_slices((dummy_features, dummy_labels))
    
    # 배치 설정 및 반복
    return dataset.batch(batch_size).repeat()


# 분산 학습 스텝 정의
@tf.function
def distributed_train_step(strategy, model, dist_inputs, optimizer):
    """분산 학습을 위한 스텝 함수"""
    
    def train_step(inputs):
        features, labels = inputs
        
        with tf.GradientTape() as tape:
            # 모델 호출
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


# Whisper 학습 함수
def train_whisper(strategy, model_type="small", num_epochs=1, learning_rate=1e-4):
    """Whisper 모델 학습 함수"""
    with strategy.scope():
        # 모델 생성
        model = create_whisper_model(model_type=model_type)
        
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
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        for _ in range(MAX_ITERATIONS):
            # 분산 데이터셋에서 배치 가져오기
            try:
                inputs = next(iterator)
            except StopIteration:
                iterator = iter(dist_dataset)
                inputs = next(iterator)
            
            # 현재 시간 기록
            step_start = time.time()
            
            # 분산 학습 스텝 실행
            loss = distributed_train_step(strategy, model, inputs, optimizer)
            
            # 스텝 완료 시간
            step_end = time.time()
            step_duration = step_end - step_start
            elapsed = step_end - start_time
            
            # 로깅
            print(f"Step {step}, Loss: {loss.numpy():.4f}, Time: {time.strftime('%H:%M:%S')} (경과: {elapsed:.2f}초, 스텝 시간: {step_duration:.2f}초)")
            
            step += 1
        
        # 에포크 종료 후 체크포인트 저장
        checkpoint.save(os.path.join(checkpoint_dir, f"whisper_{model_type}_epoch_{epoch+1}"))
    
    return model


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
    print("Whisper-small 분산 학습 시작...")
    
    # 네트워크 및 GPU 모니터링 시작
    os.system('sh /workspace/network.sh &')  # network profile
    os.system('sh /workspace/gpu.sh &')  # gpu profile
    print('''
========================
network profile started!
========================''')
    
    # JCT 측정 시작
    start_time = time.time()
    
    # 모델 학습 실행
    model = train_whisper(strategy, model_type="small")
    
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
    model_path = os.path.join(CACHE_DIR, "whisper_small_model")
    model.save_weights(model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")


if __name__ == "__main__":
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='Whisper-small Distributed Speech Recognition')
    parser.add_argument('--num_batches', type=int, default=40, help='num_batches per replica, default is set 40')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size per replica, default is set 1')
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

    print(f'batch size per replica: {BATCH_SIZE_PER_REPLICA}, global batch size: {GLOBAL_BATCH_SIZE}')
    print(f'num_batches: {MAX_ITERATIONS}')
    
    main(strategy)