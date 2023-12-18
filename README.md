# Data_Free_Quantization

You can access the project report [here](https://dropsu.sorbonne-universite.fr/s/EJHR4Ytt3g3CHrq).

This README provides instructions on how to set up your environment for the "Data_Free_Quantization" project. We'll be using virtualenv to create an isolated Python environment and conda to manage the dependencies.

## Environment Setup
### Installing virtualenv

First, you need to install the virtualenv extension. You can do this by executing the following command in your Anaconda prompt:

```bash
pip install virtualenv
```

### Creating a Virtual Environment
Once virtualenv is installed, create a Python virtual environment. For this project, we are using Python 3.8. To create your environment, run:

```bash
virtualenv myenv --python=python3.8
```
### Activating the Virtual Environment
After creating the virtual environment, you need to activate it. Use the appropriate command based on your operating system.

For Windows:

```bash
myenv\Scripts\activate
```

For Linux:

```bash
source myenv/bin/activate
```

### Installing Dependencies

To run the code for the "Data_Free_Quantization" project, you need to install specific dependencies.

- Navigate to the directory containing the Requirements.txt file.
- Run the following command:

```bash
conda install --file Requirements.txt
```

## Running the Main Function
### Executing `main_dfq.py`

The core functionality of the "Data_Free_Quantization" project is encapsulated in main_dfq.py. This script performs Data-Free Quantization (DFQ), visualizes the weights at each step, generates relevant plots, and logs the results.

To execute this script, follow these steps:

- Ensure that your virtual environment is activated and that all dependencies are installed.

- Navigate to the directory where main_dfq.py is located.

- Run the script using the following command:
  
```bash
python main_dfq.py
```
## Understanding the Output

### DFQ Process
The script will execute the Data-Free Quantization process, and you'll see progress logs in the console.

### Plot Generation
The script will automatically generate and save plots illustrating various metrics and aspects of the DFQ process. These plots help in visualizing the performance and behavior of the model throughout the quantization process.

### Logging Results
Detailed logs and results of the process will be saved, providing insights into the performance and outcomes of the quantization.

### ImageNet Validation dataset available in the following link : https://drive.google.com/file/d/1Zo0LgHWhFiVheeC21Kfvuv1B9M_tuT_8/view?usp=sharing

### Downloading and Extracting the ImageNet Training Dataset

To set up the ImageNet training dataset, follow these steps:

### 1. Download the Dataset
Use the following command to download the ImageNet validation dataset:

```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```
2. Extract the Dataset
Execute the following commands to extract the validation data and organize it into subfolders:

### Create a directory for validation data
```bash
mkdir imagenet/val
```
### Move the downloaded .tar file to the validation directory
```bash
mv ILSVRC2012_img_val.tar imagenet/val/
```
### Change to the validation directory
```bash
cd imagenet/val
```
### Extract the .tar file
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

### Quantization Bias Correction

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

