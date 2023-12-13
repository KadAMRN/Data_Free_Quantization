# Data_Free_Quantization

You can access the project report [here](https://www.overleaf.com/read/bhjhjzpwshqx#2d96d5).

## ImageNet Validation dataset available in the following link : https://drive.google.com/file/d/1Zo0LgHWhFiVheeC21Kfvuv1B9M_tuT_8/view?usp=sharing

## Downloading and Extracting the ImageNet Training Dataset

To set up the ImageNet training dataset, follow these steps:

### 1. Download the Dataset
Use the following command to download the ImageNet validation dataset:

```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```
2. Extract the Dataset
Execute the following commands to extract the validation data and organize it into subfolders:

# Create a directory for validation data
```bash
mkdir imagenet/val
```
# Move the downloaded .tar file to the validation directory
```bash
mv ILSVRC2012_img_val.tar imagenet/val/
```
# Change to the validation directory
```bash
cd imagenet/val
```
# Extract the .tar file
```bash
tar -xvf ILSVRC2012_img_val.tar
```
# Remove the compressed .tar file after extraction
rm -f ILSVRC2012_img_val.tar
```
3. Organize the Dataset
To organize the dataset into class directories, use the script provided by Soumith Chintala. Run the following command:

```bash
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```
This script creates directories for each class and moves the images into their corresponding directories. Your validation directory structure will look like this:

```bash
imagenet/val/
├── n01440764
│   ├── ILSVRC2012_val_00000293.JPEG
│   ├── ILSVRC2012_val_00002138.JPEG
│   ├── ......
├── ......
```

## Quantization Bias Correction

Dans le cadre de la quantification des DNN, une étape cruciale est la correction du biais de quantification. Cette correction est nécessaire pour atténuer les impacts négatifs que la quantification peut avoir sur la précision des modèles. La quantification, bien qu'utile pour réduire la taille du modèle et accélérer l'inférence, peut introduire des erreurs et des distorsions dans les poids et les activations du modèle. La section suivante présente les outils et méthodes que nous avons implémentés pour effectuer une correction efficace du biais de quantification, garantissant ainsi que nos modèles quantifiés conservent une performance optimale.

### clip_weight
La fonction `clip_weight` est utilisée pour contrôler les valeurs des poids dans les DNN, en les limitant à un intervalle spécifié. C'est une étape clé pour la stabilisation des modèles et leur préparation à la quantification.

### quantize_error
La fonction `quantize_error` permet de mesurer l'erreur introduite par la quantification des paramètres dans un modèle. Elle applique la quantification uniforme au paramètre d'origine, puis calcule l'erreur entre le paramètre quantifié et le paramètre original. Cette erreur est ajustée en fonction des activations préalables de la couche, afin de corriger tout biais introduit par la quantification. Elle est donc essentielle pour évaluer la justesse de la quantification dans le contexte des réseaux de neurones.

### quantize
La fonction `quantize` applique une quantification uniforme à un tenseur. Elle joue un rôle central dans la quantification des DNN,  elle inclut des classes pour la quantification de poids, d'activations et de biais dans les couches convolutionnelles et linéaires. De plus, elle fournit des outils pour mesurer dynamiquement les statistiques des activations pendant l'entraînement permettant une représentation précise et efficace des poids et des activations.

### UniformQuantize
`UniformQuantize` est une classe implémentant la quantification uniforme. Elle est fondamentale pour la quantification des réseaux neuronaux, offrant un contrôle précis sur la manière dont les valeurs sont mappées à un ensemble discret.

### QuantMeasure
`QuantMeasure` est un module PyTorch pour la quantification dynamique des activations dans un réseau NN. Il enregistre les valeurs minimales et maximales des activations, facilitant une quantification adaptative et précise en temps réel.

### bias_correction
La fonction `bias_correction` est conçue pour ajuster le biais dans les couches d'un DNN après un processus de quantification. Cette fonction est essentielle pour compenser les distorsions introduites par la réduction de la précision des poids. Elle opère en calculant l'espérance des activations et en ajustant les biais des couches cibles, notamment `nn.Conv2d` et `nn.Linear`. Ce processus aide à maintenir la performance globale du modèle en dépit des contraintes imposées par la quantification.

