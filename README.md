# De la classification binaire à la génération d'images : une première approche de l'intelligence artificielle

*This repository documents a personal research project conducted between February and March 2026, exploring the theoretical foundations and practical implementation of machine learning models - from classical statistical classifiers to state-of-the-art diffusion-based generative models.*

---

## 🇫🇷 Présentation

Ce projet est conçu comme un carnet de bord scientifique en quatre chapitres, présentant des méthodes de machine learning et de deep learning avec une difficulté croissante. L'objectif était de partir d'algorithmes statistiques simples pour progresser vers les architectures les plus récentes du Deep Learning (CNN, DDPM, RLHF). 

Chaque chapitre a été rédigé à partir de cours universitaires, de recherches personnelles et de l'étude de publications académiques. L'accent est mis sur la **compréhension mathématique** (dérivations des équations) et l'**implémentation "from scratch"** pour fonder les performances sur un fonctionnement pratique.

---

## Contenu

| Chapitre | Thématique | Concepts clés |
| :--- | :--- | :--- |
| **I** | Classification binaire | Sentiment Analysis — TF-IDF, régression logistique, SVM, Naive Bayes |
| **II** | Classification multiclasse | Dataset Iris — analyse statistique, KNN, arbres de décision |
| **III** | Classification d'images | Google QuickDraw! — MLP et CNN implémentés avec NumPy |
| **IV** | Génération d'images | MNIST/CIFAR-10/Stable-Diffusion — DDPM, U-Net, Attention, Classifier-Free Guidance, DDPO (RLHF) |

---

## Approche

Le chapitre I compare la SVM, la régression logistique et Naive Bayes appliqués à l'analyse de sentiment, en dérivant chaque modèle depuis ses fondements statistiques et géométriques. La première partie du chapitre introduit les concepts de NLP et en particulier la représentation TF-IDF, avant d'évaluer et de comparer les modèles sur un corpus réel.

Le chapitre II étend ce cadre au cas multiclasse sur le dataset Iris, en introduisant KNN et les arbres de décision. L'accent est mis sur l'analyse exploratoire des données, avec de nombreuses visualisations des features et des frontières de décision, afin que chaque résultat numérique trouve son explication dans une observation préalable.

Le chapitre III présente les limites des modèles linéaires et des MLP en vision par ordinateur, et motive l'introduction des réseaux convolutionnels. L'architecture complète du CNN, y compris la rétropropagation, est implémentée en NumPy sans framework de différentiation automatique, en traitant explicitement chaque opération du graphe de calcul. 

Le chapitre IV dérive le DDPM depuis l'ELBO, étend le modèle au conditionnement par classe via CFG, puis construit le fine-tuning par DDPO depuis les estimateurs de gradient de politique jusqu'au clipping PPO. Le code est volontairement maintenu aussi proche que possible des notations des papiers de référence.

---

## Installation
```bash
git clone https://github.com/votre-utilisateur/nom-du-repo.git
cd nom-du-repo
python -m venv venv
source venv/bin/activate  # (ou venv\Scripts\activate sur Windows)
pip install -r requirements.txt
```

---

## Références

* **LeCun et al. (1998)** — *Gradient-based learning applied to document recognition*. [HAL:hal-03926082](https://hal.science/hal-03926082v1/document)
* **Ho et al. (2020)** — *Denoising Diffusion Probabilistic Models*. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
* **Ho & Salimans (2022)** — *Classifier-Free Diffusion Guidance*. [arXiv:2207.12598](https://arxiv.org/pdf/2207.12598)
* **Schulman et al. (2017)** — *Proximal Policy Optimization Algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
* **Black et al. (2023)** — *Training Diffusion Models with Reinforcement Learning* (DDPO). [arXiv:2305.13301](https://arxiv.org/abs/2305.13301)

---

## Note

Ce projet est le fruit d'une **auto-formation** conduite en parallèle d'un cursus académique. 

Étant encore en phase d'apprentissage, certaines techniques peuvent être présentées de manière incomplète. Si vous avez des recommandations de lectures supplémentaires, ou si vous remarquez une incohérence, **n'hésitez surtout pas à me contacter**, tout retour représente toujours une vraie opportunité de progresser.