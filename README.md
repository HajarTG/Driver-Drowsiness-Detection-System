#  Driver Drowsiness Detection System

Système intelligent de détection de la somnolence au volant basé sur le Deep Learning et la vision par ordinateur.


##  Aperçu
Ce projet vise à améliorer la sécurité routière en détectant automatiquement l'état de somnolence d'un conducteur à partir d'une webcam. Deux approches sont comparées :
- **CNN from Scratch** (9.8M paramètres)
 <img width="1571" height="874" alt="image" src="https://github.com/user-attachments/assets/0efd2518-37aa-45ff-9dab-eb54608bbe01" />

- **MobileNetV2 + Transfer Learning** (2.4M paramètres)
<img width="1392" height="768" alt="image" src="https://github.com/user-attachments/assets/a7bc8500-4167-47f2-8b6e-bebc9a465125" />



##  Résultats

| Modèle | Accuracy | Recall (Sleepy) | Paramètres |
|--------|----------|-----------------|------------|
| CNN from Scratch | **98.99%** | 0.98 | 9.8M |
| MobileNetV2 | **97.01%** | 0.94 | 2.4M |

## Démo
<img width="1911" height="1003" alt="image" src="https://github.com/user-attachments/assets/67e59435-f830-4760-982e-f170b04301c0" />

<img width="1620" height="816" alt="image" src="https://github.com/user-attachments/assets/c3d79664-1136-445f-8a36-b17474fe00cf" />

##  Installation
```bash
git clone https://github.com/votre-username/driver-drowsiness-detection.git
cd driver-drowsiness-detection
pip install -r requirements.txt
python app/real_time_monitoring.py --model mobilenetv2 --threshold 0.7
