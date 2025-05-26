# Gerekli kütüphaneler
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import gc  # garbage collection için
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Psutil import (opsiyonel)
try:
    import psutil
except ImportError:
    psutil = None

# GPU ayarları - Memory growth etkinleştir
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"{len(gpus)} GPU bulundu ve bellek artışı etkinleştirildi.")
    except RuntimeError as e:
        print(f"GPU ayarlanırken hata: {e}")
else:
    print("GPU bulunamadı. CPU üzerinde çalışacak.")

# Sınıf isimleri sözlüğü
classes = {
    0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)',
    9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection',
    12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Veh > 3.5 tons prohibited',
    17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right',
    21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right',
    25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing',
    29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing',
    32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead',
    35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right',
    39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing veh > 3.5 tons'
}

# Parametreler - Enhanced V4 with Gradient Clipping
IMG_HEIGHT = 32  
IMG_WIDTH = 32   
CHANNELS = 3
NUM_CLASSES = 43
BATCH_SIZE = 8  
EPOCHS = 20      # 30'dan 20'ye düşürüldü (yüksek LR ile daha hızlı convergence)
ROUTING_ITERATIONS = 3
SEED = 42

# Random seed ayarları
np.random.seed(SEED)
tf.random.set_seed(SEED)

def squash(vectors, axis=-1):
    """
    CapsuleNet'in squashing fonksiyonu
    Vektör uzunluğunu [0,1) aralığında tutar
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

class CapsuleLayer(layers.Layer):
    """
    Enhanced CapsuleNet layer
    """
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]
        
        # 3D transformasyon matrisi: [input_dim_capsule, num_capsule, dim_capsule]
        self.W = self.add_weight(
            shape=[self.input_dim_capsule, self.num_capsule, self.dim_capsule],
            initializer='glorot_uniform',
            name='W',
            trainable=True
        )
        super(CapsuleLayer, self).build(input_shape)

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        
        # inputs shape: [batch_size, input_num_capsule, input_dim_capsule]
        # W shape: [input_dim_capsule, num_capsule, dim_capsule]
        # Use einsum for transformation: [batch_size, input_num_capsule, num_capsule, dim_capsule]
        inputs_hat = tf.einsum('bij,jkl->bikl', inputs, self.W)
        
        # Initialize coupling coefficients
        b = tf.zeros([batch_size, self.input_num_capsule, self.num_capsule])
        
        # Dynamic routing
        for i in range(self.routings):
            # Softmax on coupling coefficients
            c = tf.nn.softmax(b, axis=2)
            
            # Weighted sum: [batch_size, num_capsule, dim_capsule]
            s = tf.reduce_sum(c[..., None] * inputs_hat, axis=1)
            
            # Squash
            outputs = squash(s, axis=-1)
            
            if i < self.routings - 1:
                # Update coupling coefficients
                agreement = tf.reduce_sum(inputs_hat * outputs[:, None, :, :], axis=-1)
                b = b + agreement
        
        return outputs

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.num_capsule, self.dim_capsule)

class PrimaryCapsule(layers.Layer):
    """
    Enhanced Primary Capsules layer
    """
    def __init__(self, dim_capsule, n_channels, kernel_size, strides, padding, **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.dim_capsule = dim_capsule
        self.n_channels = n_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding

    def build(self, input_shape):
        self.conv2d = layers.Conv2D(
            filters=self.dim_capsule * self.n_channels,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            activation='relu'
        )
        super(PrimaryCapsule, self).build(input_shape)

    def call(self, inputs, training=None):
        output = self.conv2d(inputs)
        batch_size = tf.shape(output)[0]
        
        # Reshape to create capsules
        # [batch_size, height, width, n_channels * dim_capsule] -> [batch_size, n_capsules, dim_capsule]
        height = tf.shape(output)[1]
        width = tf.shape(output)[2]
        
        # Total number of capsules
        total_capsules = height * width * self.n_channels
        
        # Reshape to [batch_size, total_capsules, dim_capsule]
        outputs = tf.reshape(output, [batch_size, total_capsules, self.dim_capsule])
        
        return squash(outputs)

    def compute_output_shape(self, input_shape):
        if self.padding == 'valid':
            new_height = (input_shape[1] - self.kernel_size) // self.strides + 1
            new_width = (input_shape[2] - self.kernel_size) // self.strides + 1
        else:
            new_height = input_shape[1] // self.strides
            new_width = input_shape[2] // self.strides
            
        total_capsules = new_height * new_width * self.n_channels
        return (input_shape[0], total_capsules, self.dim_capsule)

def margin_loss(y_true, y_pred):
    """
    Margin loss for CapsuleNet
    """
    T = y_true
    L = T * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - T) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

def create_capsule_model():
    """
    Enhanced GTSRB CapsuleNet modeli V4 - Gradient Clipping Stabilized
    """
    input_image = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    
    # Layer 1: Enhanced Conv2D layers for better feature extraction
    conv1 = layers.Conv2D(24, kernel_size=3, strides=1, padding='valid', activation='relu')(input_image)  # 16->24
    conv2 = layers.Conv2D(48, kernel_size=3, strides=1, padding='valid', activation='relu')(conv1)  # 32->48
    conv3 = layers.Conv2D(64, kernel_size=3, strides=1, padding='valid', activation='relu')(conv2)  # Yeni layer
    
    # Layer 2: Enhanced Primary Capsules
    primary_capsules = PrimaryCapsule(
        dim_capsule=6, n_channels=4,  # 4->6 dim, 2->4 channels
        kernel_size=5, strides=2, padding='valid'
    )(conv3)
    
    # Layer 3: DigitCaps - Enhanced
    digitcaps = CapsuleLayer(
        num_capsule=NUM_CLASSES, 
        dim_capsule=8,  # 6'dan 8'e arttırıldı
        routings=ROUTING_ITERATIONS
    )(primary_capsules)
    
    # Layer 4: Calculate length of each capsule
    out_caps = layers.Lambda(
        lambda x: K.sqrt(K.sum(K.square(x), 2)), 
        output_shape=(NUM_CLASSES,)
    )(digitcaps)
    
    # Enhanced decoder
    y = layers.Input(shape=(NUM_CLASSES,))
    
    # Mask digitcaps
    def mask_digitcaps(inputs):
        digitcaps_input, mask_input = inputs
        mask_expanded = K.expand_dims(mask_input, -1)
        masked = digitcaps_input * mask_expanded
        return K.reshape(masked, [-1, NUM_CLASSES * 8])  # 6'dan 8'e
    
    masked_by_y = layers.Lambda(mask_digitcaps)([digitcaps, y])
    masked = layers.Lambda(mask_digitcaps)([digitcaps, out_caps])
    
    # Enhanced decoder
    decoder_input = layers.Input(shape=(NUM_CLASSES * 8,))  # 6'dan 8'e
    decoder_hidden1 = layers.Dense(128, activation='relu')(decoder_input)  # 64'ten 128'e
    decoder_hidden2 = layers.Dense(256, activation='relu')(decoder_hidden1)  # Yeni layer
    decoder_output = layers.Dense(IMG_HEIGHT * IMG_WIDTH * CHANNELS, activation='sigmoid')(decoder_hidden2)
    decoder_reshaped = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, CHANNELS))(decoder_output)
    
    decoder = models.Model(decoder_input, decoder_reshaped)
    
    # Training model
    train_model = models.Model([input_image, y], [out_caps, decoder(masked_by_y)])
    
    # Evaluation model
    eval_model = models.Model(input_image, out_caps)
    
    return train_model, eval_model

class GTSRBCapsuleNetV4Classifier:
    def __init__(self):
        self.train_model = None
        self.eval_model = None
        self.history = None
        
    def load_data_from_csv(self, train_csv_path, test_csv_path, dataset_path):
        """CSV dosyalarından veri yükler - Memory optimized"""
        print("Eğitim verisi yükleniyor...")
        train_df = pd.read_csv(train_csv_path)
        print("Test verisi yükleniyor...")
        test_df = pd.read_csv(test_csv_path)
        
        # Eğitim verilerini batch halinde yükle
        X_train = []
        y_train = []
        
        print(f"Toplam {len(train_df)} eğitim görüntüsü yükleniyor...")
        
        # Memory efficient loading
        batch_count = 0
        for idx, row in train_df.iterrows():
            img_path = os.path.join(dataset_path, row['Path'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    X_train.append(img)
                    y_train.append(row['ClassId'])
                    batch_count += 1
            
            if batch_count % 2000 == 0:
                print(f"Eğitim: {batch_count} görüntü yüklendi")
                # Garbage collection to free memory
                gc.collect()
        
        # Test verilerini yükle
        X_test = []
        y_test = []
        
        print(f"Toplam {len(test_df)} test görüntüsü yükleniyor...")
        batch_count = 0
        for idx, row in test_df.iterrows():
            img_path = os.path.join(dataset_path, row['Path'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                    X_test.append(img)
                    y_test.append(row['ClassId'])
                    batch_count += 1
            
            if batch_count % 1000 == 0:
                print(f"Test: {batch_count} görüntü yüklendi")
                gc.collect()
        
        # NumPy dizilerine dönüştür ve normalize et
        print("Veri işleniyor...")
        X_train = np.array(X_train, dtype=np.float32) / 255.0
        y_train = np.array(y_train)
        X_test = np.array(X_test, dtype=np.float32) / 255.0
        y_test = np.array(y_test)
        
        # One-hot encoding
        y_train_cat = to_categorical(y_train, NUM_CLASSES)
        y_test_cat = to_categorical(y_test, NUM_CLASSES)
        
        print(f"Eğitim veri şekli: {X_train.shape}")
        print(f"Test veri şekli: {X_test.shape}")
        print(f"Eğitim etiket şekli: {y_train_cat.shape}")
        print(f"Test etiket şekli: {y_test_cat.shape}")
        
        # Memory usage
        if psutil:
            memory_usage = psutil.virtual_memory().percent
            print(f"Memory kullanımı: {memory_usage:.1f}%")
        else:
            print("Memory kullanımı: psutil modülü bulunamadı, gösterilemiyor")
        
        return X_train, y_train_cat, X_test, y_test_cat, y_train, y_test
    
    def create_and_compile_model(self):
        """CapsuleNet V4 modelini oluşturur ve derler - Gradient Clipping ile"""
        print("Enhanced CapsuleNet V4 modeli oluşturuluyor (Gradient Clipping Stabilized)...")
        
        self.train_model, self.eval_model = create_capsule_model()
        
        # Training modeli için compile - Gradient explosion'a karşı önlemler
        self.train_model.compile(
            optimizer=Adam(
                learning_rate=0.001,   # LR restored: gradient clipping ile yeterli öğrenme için
                clipnorm=1.0          # Gradient norm clipping (Keras restriction)
            ),
            loss=[margin_loss, 'mse'],
            loss_weights=[1., 0.392],
            metrics={'lambda': 'accuracy'}
        )
        
        # Evaluation modeli için compile - Aynı optimizer ayarları
        self.eval_model.compile(
            optimizer=Adam(
                learning_rate=0.001,   # LR restored: gradient clipping ile yeterli öğrenme için
                clipnorm=1.0          # Gradient norm clipping (Keras restriction)
            ),
            loss=margin_loss,
            metrics=['accuracy']
        )
        
        print("Enhanced CapsuleNet V4 Model Özeti:")
        print("="*50)
        self.eval_model.summary()
        
        return self.train_model, self.eval_model
    
    def train_model_func(self, X_train, y_train, X_test, y_test):
        """CapsuleNet V4 modelini eğitir - Gradient explosion'a karşı gelişmiş callbacks"""
        callbacks = [
            EarlyStopping(
                monitor='val_lambda_accuracy', 
                patience=8,   # 10'dan 8'e düşürüldü (yüksek LR için daha hızlı convergence)
                restore_best_weights=True, 
                mode='max'
            ),
            ReduceLROnPlateau(
                monitor='val_loss', 
                factor=0.5,        # 0.3'ten 0.5'e arttırıldı (daha yumuşak azaltma)
                patience=4,        # 5'ten 4'e düşürüldü
                min_lr=1e-7,       # 1e-8'den 1e-7'e arttırıldı
                verbose=1
            )
        ]
        
        print("Enhanced CapsuleNet V4 eğitimi başlıyor...")
        print(f"Learning Rate: 0.001 (Restored - Gradient clipping ile yeterli öğrenme)")
        print(f"Gradient Clipping: clipnorm=1.0 (Keras restriction)")
        print(f"Batch Size: {BATCH_SIZE} (RAM optimized)")
        
        self.history = self.train_model.fit(
            [X_train, y_train], [y_train, X_train],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=([X_test, y_test], [y_test, X_test]),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test_cat, y_test_orig):
        """V4 Modeli değerlendirir"""
        print("Model V4 değerlendiriliyor...")
        
        # Tahminleri al
        y_pred_proba = self.eval_model.predict(X_test, batch_size=BATCH_SIZE)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Doğruluk hesapla
        test_loss, test_accuracy = self.eval_model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"\nCapsuleNet V4 Test Doğruluğu: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Kaybı: {test_loss:.4f}")
        
        # Unique sınıfları bul
        unique_classes = np.unique(np.concatenate([y_test_orig, y_pred]))
        
        # Confusion Matrix (yüzdelik)
        cm = confusion_matrix(y_test_orig, y_pred, labels=unique_classes)
        
        # Yüzdelik confusion matrix hesapla (her sütun için normalize et)
        # Her sütun %100 olmalı - tahmin edilen sınıf için gerçek dağılım
        cm_percentage = np.zeros_like(cm, dtype=float)
        for j in range(cm.shape[1]):  # Her sütun için (tahmin edilen sınıf)
            if cm[:, j].sum() > 0:
                cm_percentage[:, j] = cm[:, j] / cm[:, j].sum() * 100
        
        # Sınıf isimlerini kısalt (confusion matrix için)
        class_names_short = []
        for class_id in unique_classes:
            if class_id in classes:
                name = classes[class_id]
                # İsmi daha çok kısalt (ilk 10 karakter)
                if len(name) > 10:
                    name = name[:8] + ".."
                class_names_short.append(f"{class_id}: {name}")
            else:
                class_names_short.append(f"C{class_id}")
        
        # Enhanced confusion matrix görselleştirmesi - V4
        plt.figure(figsize=(20, 16))
        
        # Heatmap with enhanced spacing and readability
        ax = sns.heatmap(cm_percentage, 
                        annot=True, 
                        fmt='.1f', 
                        cmap='Blues',
                        xticklabels=class_names_short,
                        yticklabels=class_names_short,
                        cbar_kws={'label': 'Doğruluk Yüzdesi (%)'},
                        square=True,  # Kare hücreler
                        linewidths=0.5,  # Hücre arası çizgiler
                        linecolor='white',  # Çizgi rengi
                        annot_kws={'size': 8})  # Annotation boyutu
        
        # Enhanced title and labels
        plt.title('CapsuleNet V4 - LR Restored + Gradient Clipped Confusion Matrix (LR=0.001)', 
                 fontsize=18, pad=30, fontweight='bold')
        plt.xlabel('Tahmin Edilen Sınıf', fontsize=14, fontweight='bold')
        plt.ylabel('Gerçek Sınıf', fontsize=14, fontweight='bold')
        
        # Enhanced tick parameters for better readability
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        # Adjust layout for better spacing
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(bottom=0.15, left=0.15)
        
        # Save with high quality - V4 LR Restored naming
        plt.savefig('capsulenet_v4_lr_restored_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Sınıflandırma raporu
        print("\nDetaylı Sınıflandırma Raporu V4:")
        print("="*50)
        report = classification_report(y_test_orig, y_pred, labels=unique_classes,
                                     target_names=[classes.get(i, f'Class {i}') for i in unique_classes],
                                     output_dict=True, zero_division=0)
        
        # Text dosyasına sonuçları kaydet - V4
        results_text = []
        results_text.append("GTSRB CapsuleNet V4 - LR Restored + Gradient Clipping Sonuçları")
        results_text.append("=" * 65)
        results_text.append(f"Görüntü Boyutu: {IMG_HEIGHT}x{IMG_WIDTH}")
        results_text.append(f"Batch Size: {BATCH_SIZE}")
        results_text.append(f"Epochs: {EPOCHS}")
        results_text.append(f"Learning Rate: 0.001 (Restored - Gradient clipping ile yeterli öğrenme)")
        results_text.append(f"Gradient Clipping: clipnorm=1.0 (Keras restriction)")
        results_text.append(f"Model Complexity: Enhanced (24-48-64 Conv, 6x4 PrimaryCaps, 8D DigitCaps)")
        results_text.append(f"Test Doğruluğu: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        results_text.append(f"Test Kaybı: {test_loss:.4f}")
        results_text.append("\nSınıf Bazında Metrikler:")
        results_text.append("-" * 60)
        
        # Sınıf bazında metrikleri yazdır
        for class_id in unique_classes:
            class_name = classes.get(class_id, f'Class {class_id}')
            if class_name in report:
                precision = report[class_name]['precision']
                recall = report[class_name]['recall']
                f1_score = report[class_name]['f1-score']
                support = report[class_name]['support']
                
                print(f"Sınıf {class_id:2d} ({class_name}): Precision={precision:.3f}, Recall={recall:.3f}, "
                      f"F1-Score={f1_score:.3f}, Support={int(support):4d}")
                
                results_text.append(f"Sınıf {class_id:2d} ({class_name}):")
                results_text.append(f"  Precision: {precision:.3f}")
                results_text.append(f"  Recall: {recall:.3f}")
                results_text.append(f"  F1-Score: {f1_score:.3f}")
                results_text.append(f"  Support: {int(support):4d}")
                results_text.append("")
        
        # Genel metrikler
        print("\nGenel Metrikler:")
        print("="*30)
        if 'accuracy' in report:
            print(f"Doğruluk: {report['accuracy']:.4f}")
        print(f"Makro Ortalama F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Ağırlıklı Ortalama F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        results_text.append("\nGenel Metrikler:")
        results_text.append("-" * 30)
        if 'accuracy' in report:
            results_text.append(f"Doğruluk: {report['accuracy']:.4f}")
        results_text.append(f"Makro Ortalama Precision: {report['macro avg']['precision']:.4f}")
        results_text.append(f"Makro Ortalama Recall: {report['macro avg']['recall']:.4f}")
        results_text.append(f"Makro Ortalama F1-Score: {report['macro avg']['f1-score']:.4f}")
        results_text.append(f"Ağırlıklı Ortalama Precision: {report['weighted avg']['precision']:.4f}")
        results_text.append(f"Ağırlıklı Ortalama Recall: {report['weighted avg']['recall']:.4f}")
        results_text.append(f"Ağırlıklı Ortalama F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        # En iyi ve en kötü performans gösteren sınıfları bul
        class_f1_scores = []
        for class_id in unique_classes:
            class_name = classes.get(class_id, f'Class {class_id}')
            if class_name in report:
                f1_score = report[class_name]['f1-score']
                class_f1_scores.append((class_id, class_name, f1_score))
        
        class_f1_scores.sort(key=lambda x: x[2], reverse=True)
        
        results_text.append("\nEn İyi Performans Gösteren 5 Sınıf:")
        results_text.append("-" * 40)
        for i, (class_id, class_name, f1_score) in enumerate(class_f1_scores[:5]):
            results_text.append(f"{i+1}. Sınıf {class_id} ({class_name}): F1={f1_score:.3f}")
        
        results_text.append("\nEn Kötü Performans Gösteren 5 Sınıf:")
        results_text.append("-" * 40)
        for i, (class_id, class_name, f1_score) in enumerate(class_f1_scores[-5:]):
            results_text.append(f"{i+1}. Sınıf {class_id} ({class_name}): F1={f1_score:.3f}")
        
        # Model improvement details - V4 LR Restored
        results_text.append("\nCapsuleNet V4 LR Restored Geliştirme Detayları:")
        results_text.append("-" * 50)
        results_text.append("• Learning Rate: 0.0001 → 0.001 (Restored: gradient clipping ile yeterli öğrenme)")
        results_text.append("• Gradient Clipping: clipnorm=1.0 eklendi (Keras restriction)")
        results_text.append("• Early Stopping patience: 10 → 8 (yüksek LR için)")
        results_text.append("• ReduceLROnPlateau factor: 0.3 → 0.5 (daha yumuşak azaltma)")
        results_text.append("• Epoch sayısı: 30 → 20 (daha hızlı convergence)")
        results_text.append("• Conv layers: 16,32 → 24,48,64 (daha derin feature extraction)")
        results_text.append("• Primary capsule dim: 4 → 6, channels: 2 → 4")
        results_text.append("• Digit capsule dim: 6 → 8 (daha zengin representation)")
        results_text.append("• Decoder: 64 → 128,256 (daha güçlü reconstruction)")
        results_text.append("• Confusion matrix: Enhanced spacing ve readability")
        results_text.append("• CapsuleNet stability: Gradient clipping ile kararlılık artırıldı")
        results_text.append("• V4 LR Restored: Learning rate geri yüklendi, optimal öğrenme için")
        
        # Sonuçları text dosyasına kaydet - V4 LR Restored
        with open('capsulenet_v4_lr_restored_results.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(results_text))
        
        print(f"\nSonuçlar 'capsulenet_v4_lr_restored_results.txt' dosyasına kaydedildi.")
        
        return test_accuracy, y_pred, cm, cm_percentage
    
    def plot_training_history(self):
        """V4 Eğitim geçmişini çizer"""
        if self.history is None:
            print("Eğitim geçmişi bulunamadı.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['lambda_accuracy'], label='Eğitim Doğruluğu', linewidth=2)
        axes[0, 0].plot(self.history.history['val_lambda_accuracy'], label='Validasyon Doğruluğu', linewidth=2)
        axes[0, 0].set_title('CapsuleNet V4 Doğruluk (LR Restored=0.001, Gradient Clipping)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Doğruluk')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Total Loss
        axes[0, 1].plot(self.history.history['loss'], label='Eğitim Toplam Kayıp', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validasyon Toplam Kayıp', linewidth=2)
        axes[0, 1].set_title('V4 Toplam Kayıp', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Kayıp')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # CapsNet Loss
        axes[1, 0].plot(self.history.history['lambda_loss'], label='Eğitim CapsNet Kayıp', linewidth=2)
        axes[1, 0].plot(self.history.history['val_lambda_loss'], label='Validasyon CapsNet Kayıp', linewidth=2)
        axes[1, 0].set_title('V4 CapsNet Kayıp (Margin Loss)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Kayıp')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Decoder Loss
        decoder_keys = [key for key in self.history.history.keys() if 'functional' in key]
        if decoder_keys:
            train_decoder_key = [k for k in decoder_keys if 'val' not in k][0]
            val_decoder_key = [k for k in decoder_keys if 'val' in k][0]
            axes[1, 1].plot(self.history.history[train_decoder_key], label='Eğitim Decoder Kayıp', linewidth=2)
            axes[1, 1].plot(self.history.history[val_decoder_key], label='Validasyon Decoder Kayıp', linewidth=2)
        else:
            # Fallback to total loss if decoder loss not found
            axes[1, 1].plot(self.history.history['loss'], label='Eğitim Toplam Kayıp', linewidth=2)
            axes[1, 1].plot(self.history.history['val_loss'], label='Validasyon Toplam Kayıp', linewidth=2)
        
        axes[1, 1].set_title('V4 Decoder Kayıp (MSE)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Kayıp')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig('capsulenet_v4_lr_restored_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_models(self):
        """V4 LR Restored Modelleri kaydeder"""
        if self.eval_model is not None:
            self.eval_model.save('gtsrb_capsulenet_v4_lr_restored_eval_model.h5')
            print("Evaluation model 'gtsrb_capsulenet_v4_lr_restored_eval_model.h5' olarak kaydedildi.")
        
        if self.train_model is not None:
            self.train_model.save('gtsrb_capsulenet_v4_lr_restored_train_model.h5')
            print("Training model 'gtsrb_capsulenet_v4_lr_restored_train_model.h5' olarak kaydedildi.")

def main():
    """
    GTSRB CapsuleNet V4 Ana Fonksiyon - LR Restored + Gradient Clipping
    """
    # Veri yolları
    dataset_path = "/mnt/c/Users/abdul/Desktop/Code/GTSRB/GTSRB dataset"
    train_csv_path = os.path.join(dataset_path, "Train.csv")
    test_csv_path = os.path.join(dataset_path, "Test.csv")
    
    # Dosya kontrolü
    if not os.path.exists(train_csv_path):
        print(f"HATA: Train.csv bulunamadı: {train_csv_path}")
        return
    if not os.path.exists(test_csv_path):
        print(f"HATA: Test.csv bulunamadı: {test_csv_path}")
        return
    
    print("GTSRB CapsuleNet Sınıflandırıcısı V4 - LR Restored + Gradient Clipping")
    print("=" * 75)
    print("🚀 Learning Rate: 0.001 (Restored - gradient clipping ile yeterli öğrenme)")
    print("🛡️  Gradient Clipping: clipnorm=1.0 (Keras restriction)")
    print("💾 Batch Size: 8 (RAM optimized)")
    print("⚡ Enhanced model complexity ile optimal öğrenme")
    print("📁 V4 LR Restored: Mevcut çıktıları korumak için yeni versiyon")
    print("=" * 75)
    print(f"Görüntü Boyutu: {IMG_HEIGHT}x{IMG_WIDTH}")
    print(f"Sınıf Sayısı: {NUM_CLASSES}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Routing Iterations: {ROUTING_ITERATIONS}")
    print("Model Improvements (V4→V4 LR Restored):")
    print("- Learning rate restored: 0.0001 → 0.001 (optimal öğrenme)")
    print("- Gradient clipping maintained: clipnorm=1.0 (stability)")
    print("- Epochs optimized: 30 → 20 (faster convergence)")
    print("- Callbacks tuned for higher LR")
    print("- Separate output files (V4 LR Restored naming)")
    print("="*75)
    
    # Sınıflandırıcıyı başlat - V4
    classifier = GTSRBCapsuleNetV4Classifier()
    
    # Veriyi yükle
    X_train, y_train_cat, X_test, y_test_cat, y_train_orig, y_test_orig = classifier.load_data_from_csv(
        train_csv_path, test_csv_path, dataset_path
    )
    
    # Model oluştur ve derle
    train_model, eval_model = classifier.create_and_compile_model()
    
    # Modeli eğit
    history = classifier.train_model_func(X_train, y_train_cat, X_test, y_test_cat)
    
    # Eğitim geçmişini çiz
    classifier.plot_training_history()
    
    # Modeli değerlendir
    test_accuracy, predictions, confusion_mat, confusion_mat_percentage = classifier.evaluate_model(
        X_test, y_test_cat, y_test_orig
    )
    
    # Modelleri kaydet
    classifier.save_models()
    
    print(f"\n{'='*85}")
    print("🎉 CapsuleNet V4 LR Restored + Gradient Clipping eğitimi tamamlandı!")
    print(f"📊 Final Test Doğruluğu: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("📁 V4 LR Restored Çıktı dosyaları:")
    print("   • capsulenet_v4_lr_restored_confusion_matrix.png")
    print("   • capsulenet_v4_lr_restored_training_history.png")
    print("   • capsulenet_v4_lr_restored_results.txt")
    print("   • gtsrb_capsulenet_v4_lr_restored_eval_model.h5")
    print("   • gtsrb_capsulenet_v4_lr_restored_train_model.h5")
    print("💡 Tüm önceki çıktılar korundu, V4 LR Restored ayrı dosyalarda!")
    print(f"{'='*85}")

if __name__ == "__main__":
    main()
