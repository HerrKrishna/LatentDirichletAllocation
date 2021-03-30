# LatentDirichletAllocation
An implementation of online Latent Dirichlet Allocation.

As an explanation of the experiment, see the term paper.

To run the experiment install the necesary dependencies with conda:

    conda env create -f environment.yml
    conda activate LDA

The dataset is already included in this repository. So to run hyperparameter search and use test the best model run:

    python experiments.py configs/hp_search.yaml
    
To try other hyperaparameter combinations create a new config file. See hp_search.yaml as an example.

To recreate the dataset run:

    python preprocessing/create_dataset.py <path-to-corpus-directory>
    
This will Download big subreddit corpora from convokit. After every download, the programm will extract what we need and delete
the corpora again, in order to save disc space.
