# **Projet : Inversion de Visages (Face Swapping)**

## **Description**

Ce projet se concentre sur l'analyse et le traitement d'images, avec pour objectif principal d'inverser les visages de deux personnes sur des photos données. Les visages sont détectés, alignés, et échangés tout en maintenant une intégration visuelle naturelle grâce à des méthodes de clonage et de transformation.

---

## **Lien du dépôt GIT**

Le code source du projet est disponible sur GitHub :  
[https://github.com/Shahere/synthese_image](https://github.com/Shahere/synthese_image)

---

## **Pré-requis**

Avant de lancer les scripts, assurez-vous que votre environnement est correctement configuré :

### 1. **Python 3.12**

Assurez-vous que Python version 3.12 est installé sur votre machine. Vous pouvez vérifier la version avec la commande :

```bash
python --version
```

### 2. **Installation des dépendances**

Les dépendances nécessaires pour exécuter le script sont listées dans le fichier `requirements.txt`. Pour les installer, exécutez la commande suivante dans votre terminal :

```bash
pip install -r requirements.txt
```

---

## **Utilisation du Script**

Le fichier Python principal `swap_faces.py` contient tout le code nécessaire pour inverser les visages de deux images. Suivez les étapes ci-dessous pour exécuter le projet :

### 1. **Structure des Fichiers**

Assurez-vous que votre répertoire de travail contient les éléments suivants :

- `swap_faces.py` : Script principal pour l'inversion des visages.
- `requirements.txt` : Liste des bibliothèques nécessaires.
- Les deux images à traiter (par exemple `image1.jpg` et `image2.jpg`).

### 2. **Exécution**

Lancez le script en ligne de commande en fournissant les chemins des deux images en entrée :

```bash
python swap_faces.py "./image/1.jpeg" "./image/2.jpeg" True
```

le troisième paramètre correspond au debug. Indiquez True ou False

### 3. **Résultat**

Le script générera une image contenant les visages inversés, sauvegardée à l'emplacement spécifié avec l'option `--output`.
