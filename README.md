# Projet Machine Learning : Explication du prix de l'électricité

Ce projet de groupe a pour objectif d'expliquer le prix de l'électricité à partir de données météorologiques, énergétiques et commerciales pour deux pays européens, la France et l'Allemagne. Le but est d'expliquer la variation journalière des prix de contrats à terme sur l'électricité (futures), en fonction de différentes variables explicatives telles que la température, la consommation d'électricité, les prix des matières premières, etc.

## Contexte

Une multitude de facteurs influencent le prix de l'electricité au quotidien. Des variations locales du climat pourront à la fois affecter la production et la demande électrique par exemple. Des phénomènes à plus long terme, comme le réchauffement climatique, auront également un impact évident. Des évènements géopolitiques, comme la guerre en Ukraine, peuvent en parallèle faire bouger le prix des matières premières qui sont clefs dans la production d'électricité, sachant que chaque pays s'appuie sur un mix énergétique qui lui est propre (nucléaire, solaire, hydrolique, gaz, charbon, etc). De plus chaque pays peut importer/exporter de l'électricité avec ses voisins au travers de marchés dynamiques, comme en Europe. Ces différents élements rendent assez complexe la modélisation du prix de l'électricité par pays.

Le modèle doit expliquer la variation du prix de l'électricité à partir de ces données en utilisant un modèle de machine learning, en optimisant les performances avec la corrélation de Spearman.

## Objectifs

le but est de construire un modèle qui, à partir de ces variables explicatives, renvoie une bonne estimation de la variation journalière du prix de contrats à terme (dits futures) sur l'électricité, en France ou en Allemagne. Ces contrats permettent d'acheter (ou de vendre) une quantité donnée d'électricité à un prix fixé par le contrat et qui sera livrée à une date future spécifiée (maturité du contrat). Les futures sont donc des instruments financiers qui donnent une estimation de la valeur de l'électricité au moment de la maturité du contrat à partir des conditions actuelles du marché - ici, on se restreint à des futures à courte maturité (24h).

## Evaluation

La fonction de score (métrique) utilisée est la corrélation de Spearman entre la réponse du participant et les variations réelles du prix des futures contenues dans le jeu de données de test.

## Data

Les données d'entrée X_train et X_test représentent les même variables explicatives mais sur deux périodes de temps différentes.
La colonne ID de X_train et Y_train est identique, et de même pour les données test. Les données d'entrainement fournissent 1494 lignes, et les données de test en contiennent 654.

__X_train.csv : Données d'entrée d'entraînement__
__X_test.csv : Données d'entrée de test__
_ 35 colonnes
_ ID : Identifiant d'indexe unique, associé à un jour (DAY_ID) et un pays (COUNTRY),
_ DAY_ID : Identifiant du jour - les dates ont été annonymisées en préservant la structure des données,
COUNTRY : Identifiant du pays - DE = Allemagne, FR = France,
GAS_RET : Gaz en Europe,
COAL_RET : Charbon en Europe,
CARBON_RET : Futures sur les emissions carbone,

mesures météorologiques, de productions d'energie et de mesures d'utilisation électrique  (journalières, dans le pays x) :

x_TEMP : Temperature,
x_RAIN : Pluie,
x_WIND : Vent,
x_GAS : Gaz naturel,
x_COAL : Charbon,
x_HYDRO : Hydrolique,
x_NUCLEAR : Nucléaire,
x_SOLAR : Photovoltaïque,
x_WINDPOW : Eolienne,
x_LIGNITE : Lignite,
x_CONSUMPTON : Electricité totale consommée,
x_RESIDUAL_LOAD : Electricité consommée après utilisation des énergies renouvelables,
x_NET_IMPORT: Electricité importée depuis l'Europe,
x_NET_EXPORT: Electricité exportée vers l'Europe,
DE_FR_EXCHANGE: Electricité échangée entre Allemagne et France,
FR_DE_EXCHANGE: Electricité échangée entre France et Allemagne.

__Y_train.csv : Données de sortie d'entrainement__
2 colonnes
ID : Identifiant unique - le même que celui des données d'entrée,
TARGET : Variation journalière du prix de futures d'électricité (maturité 24h).
