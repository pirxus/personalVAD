# personalVAD
Bachelor's thesis project

## TODO
- check whether the aligned words really match the final utterances
- annotate concatenated data - randomly select the target speaker from the three
- noisify the concatenated utterances to midigate the concatenation artifacts
        * explore the MTR augmentation approach
- extract features and store them along with the generated annotations
- extract ivector for the target speaker and other ivectors for the whole utterance
- build the model and train it
        * also, maybe train a baseline VAD model with the same architecture
