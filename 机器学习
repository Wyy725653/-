import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, concatenate, Dense, Lambda, Dropout, \
    BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.model_selection import train_test_split
import zipfile
import os


def read_and_preprocess(file_path, tokenizer=None, max_len=50, is_train=True):
    queries1 = []
    queries2 = []
    labels = [] if is_train else None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if not parts:
                continue
            try:
                if is_train:
                    if len(parts) < 3: continue
                    q1, q2, label = parts[0], parts[1], parts[2]
                    labels.append(int(label))
                else:
                    if len(parts) < 2: continue
                    q1, q2 = parts[0], parts[1]

                queries1.append(q1.split())
                queries2.append(q2.split())
            except Exception as e:
                print(f"Error processing line: {line}")
                continue

    if is_train and tokenizer is None:
        tokenizer = Tokenizer(num_words=30000, oov_token='<OOV>')
        all_texts = queries1 + queries2
        tokenizer.fit_on_texts(all_texts)

    seq1 = tokenizer.texts_to_sequences(queries1)
    seq2 = tokenizer.texts_to_sequences(queries2)

    seq1 = pad_sequences(seq1, maxlen=max_len, padding='post', truncating='post')
    seq2 = pad_sequences(seq2, maxlen=max_len, padding='post', truncating='post')

    return (seq1, seq2, np.array(labels), tokenizer) if is_train else (seq1, seq2)


# 超参数配置
MAX_LEN = 50
EMBED_DIM = 300
VOCAB_SIZE = 30000

# 读取数据
print("Loading training data...")
train_seq1, train_seq2, train_labels, tokenizer = read_and_preprocess(
    'gaiic_track3_round1_train_20210228.csv',
    max_len=MAX_LEN,
    is_train=True
)

print("Loading test data...")
test_seq1, test_seq2 = read_and_preprocess(
    'gaiic_track3_round1_testB_20210317.csv',
    tokenizer=tokenizer,
    max_len=MAX_LEN,
    is_train=False
)

# 划分验证集
X_train1, X_val1, X_train2, X_val2, y_train, y_val = train_test_split(
    train_seq1, train_seq2, train_labels,
    test_size=0.15,
    random_state=42,
    stratify=train_labels
)

# 使用随机初始化的嵌入矩阵
print("Initializing random embedding matrix...")
embedding_matrix = np.random.normal(size=(VOCAB_SIZE + 1, EMBED_DIM))


def build_efficient_model():
    input1 = Input(shape=(MAX_LEN,))
    input2 = Input(shape=(MAX_LEN,))

    # 共享嵌入层（可训练）
    embedding = Embedding(
        VOCAB_SIZE + 1,
        EMBED_DIM,
        weights=[embedding_matrix],
        mask_zero=True,
        trainable=True
    )

    # 简化模型结构
    bilstm = Bidirectional(LSTM(96, return_sequences=False))

    x1 = bilstm(embedding(input1))
    x2 = bilstm(embedding(input2))

    # 关键特征交互
    diff = Lambda(lambda x: abs(x[0] - x[1]))([x1, x2])
    product = Lambda(lambda x: x[0] * x[1])([x1, x2])

    merged = concatenate([x1, x2, diff, product])

    # 分类头
    x = BatchNormalization()(merged)
    x = Dropout(0.3)(x)
    x = Dense(192, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input1, input2], outputs=output)

    # 使用更高效的优化器
    optimizer = Adam(learning_rate=3e-4)

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=[AUC(name='auc')]
    )
    return model


# 初始化模型
print("Building model...")
model = build_efficient_model()

# 回调函数
callbacks = [
    EarlyStopping(monitor='val_auc', patience=3, mode='max', verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=2, mode='max', min_lr=1e-5)
]

# 训练模型（减少轮次）
print("Training model...")
history = model.fit(
    [X_train1, X_train2], y_train,
    validation_data=([X_val1, X_val2], y_val),
    epochs=8,  # 减少训练轮次
    batch_size=1024,  # 增大批次提升训练速度
    callbacks=callbacks,
    verbose=1  # 显示进度条
)

# 生成预测
print("Generating predictions...")
test_pred = model.predict([test_seq1, test_seq2], batch_size=2048).flatten()

# 保存结果
np.savetxt('result.txt', test_pred, fmt='%.6f')
with zipfile.ZipFile('result.zip', 'w') as zipf:
    zipf.write('result.txt')

print("Training completed. Results saved to result.zip")
