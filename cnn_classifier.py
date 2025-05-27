import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Sınıf isimleri sözlüğü - CapsuleNet V4 ile uyumlu
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

class GTSRBCNNClassifier:
    def __init__(self, img_size=32, num_classes=43):
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_data_from_csv(self, train_csv_path, test_csv_path, dataset_path):
        """Load data using CSV files"""
        print("Loading training data...")
        train_df = pd.read_csv(train_csv_path)
        print("Loading test data...")
        test_df = pd.read_csv(test_csv_path)
        
        # Load training data
        X_train = []
        y_train = []
        
        for idx, row in train_df.iterrows():
            img_path = os.path.join(dataset_path, row['Path'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    X_train.append(img)
                    y_train.append(row['ClassId'])
            
            if (idx + 1) % 1000 == 0:
                print(f"Loaded {idx + 1} training images")
        
        # Load test data
        X_test = []
        y_test = []
        
        for idx, row in test_df.iterrows():
            img_path = os.path.join(dataset_path, row['Path'])
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    X_test.append(img)
                    y_test.append(row['ClassId'])
            
            if (idx + 1) % 1000 == 0:
                print(f"Loaded {idx + 1} test images")
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        print(f"Training data shape: {X_train.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test data shape: {X_test.shape}")
        print(f"Test labels shape: {y_test.shape}")
        print(f"Number of classes: {len(np.unique(y_train))}")
        
        return X_train, y_train, X_test, y_test
    
    def preprocess_data(self, X_train, y_train, X_test, y_test):
        """Preprocess the data"""
        # Normalize pixel values to [0, 1]
        X_train = X_train.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, self.num_classes)
        y_test_cat = to_categorical(y_test, self.num_classes)
        
        return X_train, y_train_cat, X_test, y_test_cat, y_train, y_test
    
    def create_model(self):
        """Create CNN model architecture"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(16, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            BatchNormalization(),
            Conv2D(16, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            # # Second Convolutional Block
            # Conv2D(64, (3, 3), activation='relu'),
            # BatchNormalization(),
            # Conv2D(64, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            # Dropout(0.25),
            
            # # Third Convolutional Block
            # Conv2D(128, (3, 3), activation='relu'),
            # BatchNormalization(),
            # Conv2D(128, (3, 3), activation='relu'),
            # MaxPooling2D(pool_size=(2, 2)),
            # Dropout(0.25),
            
            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
        """Train the CNN model"""
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, X_test, y_test_cat, y_test_orig):
        """Evaluate the model and create detailed confusion matrix - CapsuleNet V4 Style"""
        # Get predictions
        y_pred = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test_cat, verbose=0)
        print(f"\nCNN V12 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Unique sınıfları bul
        unique_classes = np.unique(np.concatenate([y_test_orig, y_pred_classes]))
        
        # Confusion Matrix (yüzdelik)
        cm = confusion_matrix(y_test_orig, y_pred_classes, labels=unique_classes)
        
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
        
        # Enhanced confusion matrix görselleştirmesi - V4 Style
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
                        linewidths=0.8,  # Hücre arası çizgiler (daha geniş)
                        linecolor='white',  # Çizgi rengi
                        annot_kws={'size': 8})  # Annotation boyutu
        
        # Enhanced title and labels
        plt.title('CNN V12 - LR Restored Style Confusion Matrix (LR=0.001)', 
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
        plt.savefig('cnn_v12_lr_restored_confusion_matrix.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Classification report
        print("\nDetailed Classification Report:")
        print("="*60)
        report = classification_report(y_test_orig, y_pred_classes, 
                                     target_names=[classes.get(i, f'Class {i}') for i in range(self.num_classes)],
                                     output_dict=True)
        
        # Print per-class metrics - Format hatası düzeltildi
        for class_id in range(self.num_classes):
            # Classification report'ta class ismi traffic sign ismi olarak gelir
            traffic_sign_name = classes.get(class_id, f'Class {class_id}')
            if traffic_sign_name in report:
                precision = report[traffic_sign_name]['precision']
                recall = report[traffic_sign_name]['recall']
                f1_score = report[traffic_sign_name]['f1-score']
                support = report[traffic_sign_name]['support']
                # Format hatası düzeltildi: int() ile support'u integer'a çevir
                print(f"Class {class_id:2d} ({traffic_sign_name}): Precision={precision:.3f}, Recall={recall:.3f}, "
                      f"F1-Score={f1_score:.3f}, Support={int(support):4d}")
        
        # Overall metrics
        print("\nOverall Metrics:")
        print("="*40)
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro Avg - Precision: {report['macro avg']['precision']:.4f}")
        print(f"Macro Avg - Recall: {report['macro avg']['recall']:.4f}")
        print(f"Macro Avg - F1-Score: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted Avg - Precision: {report['weighted avg']['precision']:.4f}")
        print(f"Weighted Avg - Recall: {report['weighted avg']['recall']:.4f}")
        print(f"Weighted Avg - F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        # Save results to text file - matching V4 LR Restored style
        results_text = []
        results_text.append("GTSRB CNN V12 - LR Restored Style Sonuçları")
        results_text.append("=" * 55)
        results_text.append(f"Görüntü Boyutu: {self.img_size}x{self.img_size}")
        results_text.append(f"Sınıf Sayısı: {self.num_classes}")
        results_text.append(f"Model Type: CNN (Convolutional Neural Network)")
        results_text.append(f"Architecture: Conv2D + BatchNorm + Dense layers")
        results_text.append(f"Test Doğruluğu: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        results_text.append(f"Test Kaybı: {test_loss:.4f}")
        results_text.append("\nSınıf Bazında Metrikler:")
        results_text.append("-" * 60)
        
        # Per-class metrics for text file - Düzeltildi
        for class_id in range(self.num_classes):
            # Classification report'ta class ismi traffic sign ismi olarak gelir
            traffic_sign_name = classes.get(class_id, f'Class {class_id}')
            if traffic_sign_name in report:
                precision = report[traffic_sign_name]['precision']
                recall = report[traffic_sign_name]['recall']
                f1_score = report[traffic_sign_name]['f1-score']
                support = report[traffic_sign_name]['support']
                
                results_text.append(f"Sınıf {class_id:2d} ({traffic_sign_name}):")
                results_text.append(f"  Precision: {precision:.3f}")
                results_text.append(f"  Recall: {recall:.3f}")
                results_text.append(f"  F1-Score: {f1_score:.3f}")
                results_text.append(f"  Support: {int(support):4d}")
                results_text.append("")
        
        # Overall metrics for text file
        results_text.append("\nGenel Metrikler:")
        results_text.append("-" * 30)
        results_text.append(f"Doğruluk: {report['accuracy']:.4f}")
        results_text.append(f"Makro Ortalama Precision: {report['macro avg']['precision']:.4f}")
        results_text.append(f"Makro Ortalama Recall: {report['macro avg']['recall']:.4f}")
        results_text.append(f"Makro Ortalama F1-Score: {report['macro avg']['f1-score']:.4f}")
        results_text.append(f"Ağırlıklı Ortalama Precision: {report['weighted avg']['precision']:.4f}")
        results_text.append(f"Ağırlıklı Ortalama Recall: {report['weighted avg']['recall']:.4f}")
        results_text.append(f"Ağırlıklı Ortalama F1-Score: {report['weighted avg']['f1-score']:.4f}")
        
        # CNN Model details
        results_text.append("\nCNN V12 Model Detayları:")
        results_text.append("-" * 35)
        results_text.append("• Architecture: Simplified CNN (single Conv block)")
        results_text.append("• Conv Layer: 16 filters, 3x3 kernel, ReLU activation")
        results_text.append("• BatchNormalization: After each Conv2D")
        results_text.append("• MaxPooling: 2x2 pool size")
        results_text.append("• Dropout: 0.25 after Conv, 0.5 after Dense")
        results_text.append("• Dense Layers: 512 → 256 → 43 (output)")
        results_text.append("• Optimizer: Adam with learning_rate=0.001")
        results_text.append("• Loss: categorical_crossentropy")
        results_text.append("• Early Stopping: val_accuracy, patience=10")
        results_text.append("• ReduceLROnPlateau: factor=0.2, patience=5")
        results_text.append("• Output style: V4 LR Restored naming convention")
        results_text.append("• Confusion Matrix: Percentage-based like CapsuleNet V4")
        
        # Save results to file
        with open('cnn_v12_lr_restored_results.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(results_text))
        
        print(f"\nSonuçlar 'cnn_v12_lr_restored_results.txt' dosyasına kaydedildi.")
        
        return test_accuracy, y_pred_classes, cm
    
    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('CNN V12 LR Restored - Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('CNN V12 LR Restored - Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('cnn_v12_lr_restored_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Define paths
    dataset_path = "/mnt/c/Users/abdul/Desktop/Code/GTSRB/GTSRB dataset"
    train_csv_path = os.path.join(dataset_path, "Train.csv")
    test_csv_path = os.path.join(dataset_path, "Test.csv")
    
    # Check if files exist
    if not os.path.exists(train_csv_path):
        print(f"Error: Train.csv not found at {train_csv_path}")
        return
    if not os.path.exists(test_csv_path):
        print(f"Error: Test.csv not found at {test_csv_path}")
        return
    
    print("GTSRB CNN V12 Classification - LR Restored Style")
    print("="*55)
    print("🚀 CNN Architecture: Simplified single Conv block design")
    print("⚡ Output naming: V4 LR Restored convention") 
    print("📁 Compatible with CapsuleNet V4 outputs")
    print("📊 Confusion Matrix: CapsuleNet V4 style (percentage-based)")
    print("🏷️  Class Names: German Traffic Sign Recognition")
    print("="*55)
    
    # Initialize classifier
    classifier = GTSRBCNNClassifier(img_size=32, num_classes=43)
    
    # Load data
    X_train, y_train, X_test, y_test = classifier.load_data_from_csv(
        train_csv_path, test_csv_path, dataset_path
    )
    
    # Preprocess data
    X_train_norm, y_train_cat, X_test_norm, y_test_cat, y_train_orig, y_test_orig = classifier.preprocess_data(
        X_train, y_train, X_test, y_test
    )
    
    # Create and compile model
    print("\nCreating CNN model...")
    model = classifier.create_model()
    print(model.summary())
    
    # Train model
    print("\nTraining model...")
    history = classifier.train_model(
        X_train_norm, y_train_cat, X_test_norm, y_test_cat, 
        epochs=50, batch_size=32
    )
    
    # Plot training history
    classifier.plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    test_accuracy, predictions, confusion_mat = classifier.evaluate_model(
        X_test_norm, y_test_cat, y_test_orig
    )
    
    # Save model
    model_save_path = 'gtsrb_cnn_v12_lr_restored_model.h5'
    classifier.model.save(model_save_path)
    print(f"\nModel saved as: {model_save_path}")
    
    print(f"\n{'='*60}")
    print("🎉 CNN V12 - LR Restored Style eğitimi tamamlandı!")
    print(f"📊 Final Test Doğruluğu: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print("📁 Çıktı dosyaları:")
    print("   • cnn_v12_lr_restored_confusion_matrix.png")
    print("   • cnn_v12_lr_restored_training_history.png")
    print("   • cnn_v12_lr_restored_results.txt")
    print("   • gtsrb_cnn_v12_lr_restored_model.h5")
    print("💡 V4 LR Restored naming convention kullanıldı!")
    print("📈 CapsuleNet V4 style confusion matrix (yüzdelik)")
    print("🏷️  German traffic sign class names included")
    print(f"{'='*60}")
    
    print("\nTraining and evaluation completed!")

if __name__ == "__main__":
    main()
