# Chest X-ray Pneumonia Classification

## Prezentare Generala

Pneumonia este o afectiune respiratorie comuna, dar serioasa, caracterizata prin inflamarea tesutului pulmonar. Diagnosticarea corecta si rapida este esentiala, deoarece intarzierile in identificarea bolii pot duce la complicatii severe. In practica medicala, radiografiile toracice reprezinta una dintre cele mai folosite metode pentru evaluarea starii plamanilor.

Acest proiect utilizeaza **deep learning** pentru a automatiza diagnosticarea pneumoniei in radiografii toracice, sprijinind procesul de diagnostic si reducand incarcarea personalului medical.

## Obiectivul Proiectului

Construirea unui model de deep learning capabil sa identifice automat prezenta pneumoniei in imaginile radiologice. Modelul utilizeaza arhitectura **VGG16** prin transfer learning, o metoda eficienta care permite reutilizarea caracteristicilor deja invatate pe setul ImageNet.

## Fluxul Proiectului

1. **Incarcarea si explorarea datelor** - Vizualizare si analiza distributiei setului de date
2. **Pregatirea imaginilor** - Redimensionare si normalizare
3. **Augmentarea datelor** - Cresterea variabilitatii pentru evitarea overfitting-ului
4. **Definirea modelului** - Construire arhitectura VGG16 cu straturi custom
5. **Antrenarea modelului** - Utilizarea callback-urilor pentru optimizare
6. **Evaluarea performantelor** - Metrici detaliate si analiza erorilor

## Dataset

- **Format**: Imagini radiologice in format JPEG
- **Clase**: 2 (NORMAL si PNEUMONIA)
- **Status**: Set de date dezechilibrat - categoria PNEUMONIA are mai multe exemple
- **Split**: 
  - Antrenament: 4,229 imagini
  - Test: 1,044 imagini

## Arhitectura Modelului

### VGG16 + Custom Layers

┌─────────────────────────────────────────┐
│ VGG16 (ImageNet Pretrained) │
│ - Blocate straturile convolutive │
│ - Reutilizare feature-uri │
└─────────────────────────────────────────┘
↓
┌─────────────────────────────────────────┐
│ GlobalAveragePooling2D │
│ - Reducere dimensionalitate │
└─────────────────────────────────────────┘
↓
┌─────────────────────────────────────────┐
│ Dense(512) + ReLU │
│ BatchNormalization │
│ Dropout(0.5) │
└─────────────────────────────────────────┘
↓
┌─────────────────────────────────────────┐
│ Dense(256) + ReLU │
│ BatchNormalization │
│ Dropout(0.3) │
└─────────────────────────────────────────┘
↓
┌─────────────────────────────────────────┐
│ Dense(1) + Sigmoid │
│ - Output: Probabilitate [0, 1] │
└─────────────────────────────────────────┘


## Performante

### Rezultate pe Setul de Test

- **Acuratete**: 92.53%
- **Precizie**: 99.72%
- **Recall**: 90.21%
- **AUC (ROC)**: 0.9944

### Metrici Detaliate

| Clasa | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Normal | 0.78 | 0.99 | 0.87 |
| Pneumonia | 1.00 | 0.90 | 0.95 |
| **Weighted Avg** | **0.94** | **0.93** | **0.93** |

## Tehnici Utilizate

### Transfer Learning
- Utilizare model VGG16 pre-antrenat pe ImageNet
- Blocarea straturilor convolutive pentru a pastra feature-uri generale
- Fine-tuning prin straturi custom dense

### Data Augmentation
- Rotatii random (±15°)
- Translatii orizontale si verticale (±10%)
- Zoom random (±10%)
- Flip orizontal random
- Efect shear

### Regularizare
- **Dropout**: Eliminare aleatorie de neuroni pentru prevenirea overfitting-ului
- **Batch Normalization**: Normalizare intrari straturi
- **Class Weights**: Ponderare inverse pentru date dezechilibrate

### Optimizare
- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Binary Crossentropy
- **EarlyStopping**: Oprire antrenament daca loss nu se imbunatateste
- **ReduceLROnPlateau**: Reducere learning rate adaptiva
- **ModelCheckpoint**: Salvare model cu cea mai buna acuratete

## Instalare si Configurare

### Cerinte Preliminare
- Python 3.8+
- GPU (recomandat pentru viteza mai mare)
- pip sau conda

### Pasi de Instalare

```bash
conda create -n chestxray python=3.9 -y
conda activate chestxray
conda install tensorflow keras matplotlib seaborn scikit-learn numpy pillow opencv jupyter -c conda-forge -y
jupyter notebook
```



### Structura Directoarelor
project/
├── dataset/
│   └── chest_xray_split/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── bun1.keras                 # Model antrenat
├── train.py                   # Script antrenament
├── evaluate.py                # Script evaluare
├── requirements.txt           # Dependinte Python
└── README.md                  # Acest fisier

Explorarea Datelor

## Explorarea Datelor

Scriptul include vizualizari pentru:

- **Exemple de imagini**: imagini din ambele clase  
- **Distributia claselor**: numarul de imagini pe clasa in fiecare split  
- **Dimensiuni imagini**: histograme ale latimii si inaltimii  
- **Data Augmentation**: exemple ale transformarilor aplicate  

## Metrici - Explicatie

- **Acuratete**: procentajul total de predictii corecte  
- **Precizie**: procentajul predictiilor pozitive corecte (cat de sigur suntem ca pneumonia detectata este reala)  
- **Recall (Sensibilitate)**: procentajul cazurilor pozitive detectate (cat de bine identificam cazurile de pneumonie)  
- **F1-Score**: medie armonica intre precizie si recall  
- **AUC**: aria sub curba ROC - masoara capacitatea modelului de separare intre clase  

## Observatii Importante

- **Dezechilibrul claselor**: dataset-ul contine mai multe imagini cu pneumonie. Acest dezechilibru este gestionat prin:  
  - *Class weights* in antrenament  
  - Metrici multiple pentru evaluare  

- **Precizie ridicata, Recall mai scazut pentru clasa NORMAL** :  
  Modelul este mai conservator in detectarea pneumoniei.  
  Este mai sigur sa marcheze o radiografie ca NORMAL doar daca este foarte sigur, pentru a evita alarmele false.  
  Acesta este un comportament dezirabil din perspectiva medicala.  

- **Transfer Learning**: utilizarea VGG16 pre-antrenat permite:  
  - Antrenament mai rapid  
  - Rezultate mai bune cu mai putine date  
  - Utilizare redusa de memorie si GPU  

## Autori si Contributii

Proiect academic dezvoltat pentru clasificarea automata a radiografiilor toracice folosind deep learning si transfer learning cu arhitectura **VGG16**.

# Referinte si resurse
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [ImageNet Dataset](http://www.image-net.org/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Documentation](https://keras.io/)
- [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)
