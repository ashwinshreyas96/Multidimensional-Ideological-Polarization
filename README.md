# Multidimensional Ideological Polarization

GitHub Repository for the paper - [Political Partisanship and Anti-Science Attitudes in Online Discussions about COVID-19](https://arxiv.org/abs/2011.08498). This repository consists of code that perform pre-processing, embedding generation and classification of user's multi-dimensional ideological polarity.

The COVID-19 pandemic has laid bare our differences and exposed crevices in national unity. Ideological fissures on social media have fragmented the nation into seemingly distinct politico-scientific groups. In this repository, we publish the code to identify such groups on Twitter data and analyze the interplay between them. 


## Usage

### FastText Model

Tweet-ids corresponding to COVID-19 tweets collected from January 21, 2020 to July 31, 2020 have been sourced from [COVID-19-TweetIDs](https://github.com/echen102/COVID-19-TweetIDs). Owing to Twitter's policy we are restricted to sharing tweet-ids and users can rehydrate this dataset using [hydrator](https://github.com/DocNow/hydrator). Upon rehydrating the content we can start pre-processing our tweets to compute ground truth domain scores. Ensure that the folder structure of rehydrated tweets match the folder structure of tweet-ids. 

Refer to [twitter-locations-us-state](https://github.com/julie-jiang/twitter-locations-us-state) to perform filtering of rehydrated tweets by their location. 

To extract domains and users from rehyrdated tweets:

```
python extract_domains.py <path_to_rehydrated_tweets> #extracts domains and users from rehydrated tweets
```
Next, compute ideological domain scores for each of the Science, Political and Moderacy dimensions.

```
python Domain-Score-Calc.py
```
Concatenate extracted tweets over time into one .csv file and execute the following statement to generate user specific tweet embeddings. To generate embeddings you need to download the FastText Twitter bi-gram model from [Sent2Vec](https://drive.google.com/file/d/0B6VhzidiLvjSeHI4cmdQdXpTRHc/view) and place it in the models directory. 

```
python mbfc-covid-fasttext.py <path to concatenated .csv file consisting of users and tweets>
```

You are then ready to execute the prediction model as follows and the final prediction results are saved in the results folder.

```
python FastText-Pred.py
```

### LDA Model

Prepare a dataset consisting of users and the hashtags they generate over time and use the following command to conduct LDA analysis.

```
python LDA-Analysis.py <path_to_.csv containing user-hashtag data>
```

## Citation

If yopu find this code useful please cite [Political Partisanship and Anti-Science Attitudes in Online Discussions about COVID-19](https://arxiv.org/abs/2011.08498) as follows:

```
@misc{rao2020political,
      title={Political Partisanship and Anti-Science Attitudes in Online Discussions about Covid-19}, 
      author={Ashwin Rao and Fred Morstatter and Minda Hu and Emily Chen and Keith Burghardt and Emilio Ferrara and Kristina Lerman},
      year={2020},
      eprint={2011.08498},
      archivePrefix={arXiv},
      primaryClass={cs.SI}
}
```
