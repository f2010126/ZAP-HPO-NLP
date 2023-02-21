# ZAP-HPO-NLP
### Test the adaptation of the ZAP-HPO framework for the NLP domain

## Goal
This is the begining of my Master's thesis. Verifying the domain-independent nature of [ZAP-HPO](https://github.com/automl/zero-shot-automl-with-pretrained-models).
Notes and discussions are kept in [My Obsidian Vault](https://github.com/f2010126/ObsidianVaults/tree/main/Thesis)




|  Task | Goal |Estimated Duration | 
|  ----------- | ----------- |-------|
|       Choice of MetaData       | using simple features like word count won't help much in the context of NLP | Done? |
|       Choice of models       | Deep Models like BertBase, MiniBert, TinyBert | 3-4 weeks |
|      Upgrade ZAP framework       | 3.7 no longer supported,  | 2 weeks |
|   Rework framework, base model no longer AutoCV      | If benchmark is no longer just AutoNLP | Estimate |
|        Store the datasets in a central location, keep a list of what is where       | easy access | 3 days |
|        Using the embeddings      |  emeddings of the datasets and how it will be stored/used  | Estimate |
|       Define Benchmarks       | What to compare against | 3-5 days |
|       Hyperparameter space definition       | use one from the vision domain plus specific to language | 3-5 days |
|       Create the cost matrix - mini 1 model, 2datasets       | Get end to end implementation on small scale | 4-6 weeks |
|       Create the cost matrix - full      | Why | 1 week |
|       Generate the dataset for the surrogate model.      | given the cost matrix, the HPO configs of the pipelines and the dataset embeddings, generate the metadata dataset | 3-5 days |
|       Training surrogate model choice loss function     | Why | Estimate |
|       Evaluate on Benchmarks     | Why | Estimate |


Tasks like Choice of models, Hyperparameter space definition, database storage, Benchmark selection done parallel.


## Upgrade ZAP framework
- Python 3.7 to 3.9
- tensorflow 1.x to 2.x
- Smac4MF instead of BoHB (I have the code I need to use/change)
Blocked by Cuda errors, when run on cpu, gives negative score. 


## Choice of models:
### 1. Do I pretrain the models or is it just finetuning?
Training details and some HPO suggestions: https://github.com/stefan-it/europeana-bert
May need to pretrain TinyBert, MiniBERT from scratch.

### 2. Why is this needed when LLM already have good zero shot?
- pretraining is still expensive, not open to all
- models are not usually trained for german. 

## Rework framework
- Use the AutoDL workflow only for the anytime performance measurement, else for execution (creation of metadata set,) just set a timer or limit the runtime. 
- Reason for avoiding AutoDL workfow, code is old. Or just update the repo and use it.


## Generate the embeddings

### 1. What model to use? 
### 2.  How to store that and how to use it?

Idea: 
During the cost matrix generation, each pipeline generates the embedding and it is stored. another similarity matrix, visual rep of dataset similarity and final acc
During the surrogate final acc is give along with the train. 
Is there some way where the model is able to use to generate the embedding for the test dataset, compare against the rest and only use the datasets-pipeline combos nearest to it?
Any other way for the 


## Benchmarks
- Use of AutoNLP might not work since the datasets can't be downloaded.
- Need a language German benchmarks? benchmark. 
- Most benchmarks are in English. So take the muliti lingual datasets and use the DE portion.
Idea: use the same ones as DBMDZ (77.852 ± 0.60 F1)



# Challenges
1. GPU, cluster training feels slow.
2. Errors with ZAP upgrade
3. BOHB feels very slow. What options am I not setting?
4. How to use the embeddings?
