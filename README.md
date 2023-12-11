# Data_Free_Quantization

## ImageNet Validation dataset available in the following link : https://drive.google.com/file/d/1Zo0LgHWhFiVheeC21Kfvuv1B9M_tuT_8/view?usp=sharing

## Quantization Bias Correction

Dans le cadre de la quantification des DNN, une étape cruciale est la correction du biais de quantification. Cette correction est nécessaire pour atténuer les impacts négatifs que la quantification peut avoir sur la précision des modèles. La quantification, bien qu'utile pour réduire la taille du modèle et accélérer l'inférence, peut introduire des erreurs et des distorsions dans les poids et les activations du modèle. La section suivante présente les outils et méthodes que nous avons implémentés pour effectuer une correction efficace du biais de quantification, garantissant ainsi que nos modèles quantifiés conservent une performance optimale.

### clip_weight
La fonction `clip_weight` est utilisée pour contrôler les valeurs des poids dans les DNN, en les limitant à un intervalle spécifié. C'est une étape clé pour la stabilisation des modèles et leur préparation à la quantification.

### _quantize_error
La fonction `_quantize_error` permet de mesurer l'erreur introduite par la quantification des paramètres dans un modèle. Elle offre une vue détaillée de l'impact de la quantification, essentielle pour optimiser et ajuster les processus de quantification.

### quantize
La fonction `quantize` applique une quantification uniforme à un tenseur. Elle joue un rôle central dans la quantification des DNN, permettant une représentation précise et efficace des poids et des activations.

### UniformQuantize
`UniformQuantize` est une classe implémentant la quantification uniforme. Elle est fondamentale pour la quantification des réseaux neuronaux, offrant un contrôle précis sur la manière dont les valeurs sont mappées à un ensemble discret.

### QuantMeasure
`QuantMeasure` est un module PyTorch pour la quantification dynamique des activations dans un réseau NN. Il enregistre les valeurs minimales et maximales des activations, facilitant une quantification adaptative et précise.

### bias_correction
La fonction `bias_correction` est conçue pour ajuster le biais dans les couches d'un DNN après un processus de quantification. Cette fonction est essentielle pour compenser les distorsions introduites par la réduction de la précision des poids. Elle opère en calculant l'espérance des activations et en ajustant les biais des couches cibles, notamment `nn.Conv2d` et `nn.Linear`. Ce processus aide à maintenir la performance globale du modèle en dépit des contraintes imposées par la quantification.

