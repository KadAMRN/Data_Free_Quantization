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

## Downloading the Dataset
### For the classification task :

#### ImageNet Validation dataset available in the following link if you are on a Windows machine : 
https://drive.google.com/file/d/1Zo0LgHWhFiVheeC21Kfvuv1B9M_tuT_8/view?usp=sharing

#### Downloading and Extracting the ImageNet Validation Dataset for a Linux machine

To set up the ImageNet `validation dataset, follow these steps:

Use the following command to download the ImageNet validation dataset:

```bash
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```
2. Extract the Dataset
Execute the following commands to extract the validation data and organize it into subfolders:

#### Create a directory for validation data
```bash
mkdir imagenet/val
```
#### Move the downloaded .tar file to the validation directory
```bash
mv ILSVRC2012_img_val.tar imagenet/val/
```
#### Change to the validation directory
```bash
cd imagenet/val
```
#### Extract the .tar file
```bash
tar -xvf ILSVRC2012_img_val.tar
```
#### Remove the compressed .tar file after extraction
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
### For the segmentation task :
For our segmantaion task validation we use the Development Kit VOC2012 validation dataset available on the following website : http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
It comes as a compressed file that should be extracted and put into the Data_Free_Quantization main folder.

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

## Command Overview

To execute the main functionality of the Data-Free Quantization (DFQ) process with 8-bit quantization, use the following command:

```bash
python main_dfq.py --gpu --task cls --relu --equalize --absorption --quantize --correction --clip_weight --log --bits_weight 8 --bits_activation 8 --bits_bias 8
```

## Detailed Argument Explanations

- `--gpu`: Engages the GPU for computational tasks, leveraging its processing power for faster execution.

- `--task cls`: Specifies that the task is classification (`cls`). This setting configures the script for classification-related operations.

- `--relu`: Activates the ReLU (Rectified Linear Unit) function in the neural network. ReLU is a popular activation function known for its efficiency in deep learning models.

- `--equalize`: Adjusts the balance or distribution of data or weights. In quantization, this helps in optimizing the model's performance.

- `--absorption`: A technique used in the quantization process that may involve merging or simplifying certain layers or parameters for efficiency.

- `--quantize`: Triggers the process of quantization, which reduces the precision of the model's weights and activations to a lower bit representation.

- `--correction`: A post-quantization step that adjusts the model to compensate for any loss of accuracy due to reduced precision.

- `--clip_weight`: Limits the range of the model's weights. This is often used to maintain the stability and performance of the model after quantization.

- `--log`: Enables the generation of logs during the process, providing detailed insights into the operations and results of the script.

- `--bits_weight 8`: Sets the number of bits for the weights to 8. This is a key aspect of quantization, impacting the model's size and computational efficiency.

- `--bits_activation 8`: Specifies that the activations (outputs of layers) in the neural network are quantized to 8 bits.

- `--bits_bias 8`: Determines that biases in the neural network are also quantized to 8 bits, ensuring consistency in precision across different components of the model.

## Output Plots

Below is an example of the output plots generated by the script, showing the histogram of weights of the 18th layer before and after Data-Free Quantization (DFQ):

![Histogram of Weights of 18th Layer Before/After DFQ](https://i.imgur.com/24AtVXa.png)

*Figure: This histogram compares the distribution of weights before and after applying DFQ, illustrating the effect of the quantization process on the model's weights.*

## Output Plots for Quantization

Below is an example of the output plots generated by the script, which depicts the histogram of weights of the 18th layer before and after only the quantization step:

![Histogram of Weights of 18th Layer Before/After Quantization](https://i.imgur.com/f4Lo3dJ.png)

*Figure: This histogram shows the distribution of weights before and after the quantization step, highlighting the impact of this specific quantization on the model's weights.*


## Before and After Equalization Plots

These plots illustrate the weight range of different output layers before and after the equalization process, which is a part of the Data-Free Quantization steps to optimize the performance of the neural network model.

### Before Equalization
![Before Equalization](https://i.imgur.com/LphCfpb.png)

*Figure: Weight range across various output layers before the equalization process.*

### After Equalization
![After Equalization](https://i.imgur.com/KyR1ajb.png)

*Figure: Weight range across various output layers after the equalization process, demonstrating the effect of equalization.*

### After High Biases Absorbtion

<img width="842" alt="image" src="https://github.com/KadAMRN/Data_Free_Quantization/assets/87100217/4b49a4e0-3587-4008-b04f-105e64623548">

*Figure: Histograms of a Conv2D layer bias values before and after high bias absorbtion*

## Post-Quantization Output

The following figure illustrates the changes in weight distribution across various output layers after the quantization step. This step is crucial for reducing model size and computational demand without significantly impacting performance.

### After Quantization
![After Quantization](https://i.imgur.com/BzLd4H1.jpg)

*Figure: Boxplot and histogram showcasing weight distribution and frequency after quantization.*

## Post-Bias Correction Output


<img width="882" alt="image" src="https://github.com/KadAMRN/Data_Free_Quantization/assets/87100217/843c6d09-5ea0-450b-a87d-486b02022bf9">

*Figure: Histograms of original and updated post
quantization biases via bias correction of a layer for a 3-bit quantization*


*Figure: Boxplot and histogram showing weight distribution and frequency after bias correction.*

## DFQ Results log Overview

The results from the Data-Free Quantization (DFQ) process are detailed in the `dfq_result.txt` file located in the project directory. This file contains the configurations and the corresponding accuracy metric for various runs of the DFQ process.

The following table provides a comprehensive summary of the results derived from the data file dfq_result.txt using the **MobileNetV2** architecture for the **classification** task

| Architecture\precision% | FP 32 | int8 | int 16 | int 8 with int 16 bias |
| ----------------------- | ----- | ---- | ------ | ---------------------- |
| Original                | 71.82 | 0.11 | 69.28  | 0.12                   |
| ReLU+LE+BA              | 71.59 | 68.14| 70.78  | 70.27                  |
| BC                      | --    | 0.11 | 69.28  | 0.10                   |
| BC+CW                   | --    | 0.12 | 66.59  | 0.10                   |
| ReLU+LE+BA+BC           | --    | 68.12| 70.344 | 70.294                 |
| ReLU+LE+BA+BC+CW        | --    | 68.12| 70.34  | 70.29                  |

As for the **segmentation** task using the **DeeplabV3+** model as The int8 quantization resulted in an accuracy of **90.42\%**, compared to the original FP32 model, which achieved an accuracy of **92.19\%**


### Useful information (in french) concerning the quantization and the bias correction
#### Quantization Bias Correction

Dans le cadre de la quantification des DNN, une étape cruciale est la correction du biais de quantification. Cette correction est nécessaire pour atténuer les impacts négatifs que la quantification peut avoir sur la précision des modèles. La quantification, bien qu'utile pour réduire la taille du modèle et accélérer l'inférence, peut introduire des erreurs et des distorsions dans les poids et les activations du modèle. La section suivante présente les outils et méthodes que nous avons implémentés pour effectuer une correction efficace du biais de quantification, garantissant ainsi que nos modèles quantifiés conservent une performance optimale.

#### clip_weight
La fonction `clip_weight` est utilisée pour contrôler les valeurs des poids dans les DNN, en les limitant à un intervalle spécifié. C'est une étape clé pour la stabilisation des modèles et leur préparation à la quantification.

#### quantize_error
La fonction `quantize_error` permet de mesurer l'erreur introduite par la quantification des paramètres dans un modèle. Elle applique la quantification uniforme au paramètre d'origine, puis calcule l'erreur entre le paramètre quantifié et le paramètre original. Cette erreur est ajustée en fonction des activations préalables de la couche, afin de corriger tout biais introduit par la quantification. Elle est donc essentielle pour évaluer la justesse de la quantification dans le contexte des réseaux de neurones.

#### quantize
La fonction `quantize` applique une quantification uniforme à un tenseur. Elle joue un rôle central dans la quantification des DNN,  elle inclut des classes pour la quantification de poids, d'activations et de biais dans les couches convolutionnelles et linéaires. De plus, elle fournit des outils pour mesurer dynamiquement les statistiques des activations pendant l'entraînement permettant une représentation précise et efficace des poids et des activations.

#### UniformQuantize
`UniformQuantize` est une classe implémentant la quantification uniforme. Elle est fondamentale pour la quantification des réseaux neuronaux, offrant un contrôle précis sur la manière dont les valeurs sont mappées à un ensemble discret.

#### QuantMeasure
`QuantMeasure` est un module PyTorch pour la quantification dynamique des activations dans un réseau NN. Il enregistre les valeurs minimales et maximales des activations, facilitant une quantification adaptative et précise en temps réel.

#### bias_correction
La fonction `bias_correction` est conçue pour ajuster le biais dans les couches d'un DNN après un processus de quantification. Cette fonction est essentielle pour compenser les distorsions introduites par la réduction de la précision des poids. Elle opère en calculant l'espérance des activations et en ajustant les biais des couches cibles, notamment `nn.Conv2d` et `nn.Linear`. Ce processus aide à maintenir la performance globale du modèle en dépit des contraintes imposées par la quantification.

## Acknowledgment
- https://github.com/ricky40403/PyTransformer
- https://github.com/tonylins/pytorch-mobilenet-v2
- https://github.com/jfzhang95/pytorch-deeplab-xception
- https://github.com/eladhoffer/quantized.pytorch
- https://github.com/quic/aimet
