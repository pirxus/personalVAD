# personalVAD
Bachelor's thesis project


## CHECK CHECK CHECK
- audio preprocessing - volume normalization


## TODO
### training
- get familiar with metacentrum, train a model

### data prep
- look into kaldi augmentation
        - prepare scripts for utterance reverberation and augmentation
        - maybe rework the feature extraction pipeline to work with multiple
          recordings for each utterance id (augmented recordings)

- noisify the concatenated utterances to midigate the concatenation artifacts
        - explore the MTR augmentation approach

- annotate concatenated data - randomly select the target speaker from the three
- extract an enrollment d-vector for each speaker and prepare a pipeline for
  speaker verification while training

- build the model and train it
        - also, maybe train a baseline VAD model with the same architecture


## Feature extraction pipeline
1) create concatenated utterances
2) reverberate and augment the whole dataset
3) for each utterance id:
    - create the augmented versions
    - extract log-fbanks features for each recording
    - create the ground truth label array
    - randomly select one of the speakers to be the target speaker
        - for this speaker, extract the embedding
        - store the embedding and the ground truth labels into a separate folder along with
          the utterance id
    - CARE: check, whether the lengths of the augmented versions of the recordings (or their
      feature vectors respectively) match the lenght of the ground truth label array and if not,
      probably just trim them...
    - save the extracted features under different utterance id (in order to distinguish the clean
      from the augmented ones) and delete the augmented recordings if necessary
4) result: two new data folders
    a) one containing the ground truth label arrays and target speaker embeddings
    b) the second one with all the extracted features from the whole augmented dataset


## Training pipeline
1) the dataloader first creates a representation of the dataset - just collect all utterance ids..
2) select a batch of utterances and load the corresponding ground truth labels along with the
   target speaker embeddings
