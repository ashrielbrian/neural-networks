# neural-networks
Neural networks written in numpy. Current NNs:

1. __2-layer FC (basic)__
2. __3-layer FC__
  - Classifies whether an image is a cat or not.
  - Depends on: `datasets/train_catvnoncat.h5`, `datasets/test_catvnoncat.h5`, `load_dataset`
3. __RNN__
  - Character-level RNN based off @karpathy's gist (https://gist.github.com/karpathy/d4dee566867f8291f086)
  - Adagrad Optimizer
  - Depends on: `datasets/shakespearean.txt`. Can be used with `datasets/hp1.txt`
4. __LSTM__
  - Character-level shallow LSTM
  - Adam Optimizer, Gradient Checking
  - Depends on: `datasets/shakespearean.txt`. Can be used with `datasets/hp1.txt`

Please let me know if you have any questions.

The scripts were written with the purpose of informing my understanding of NNs on a fundamental level. While I have tried to keep the code clean, my priority did not lie in writing clever Python code.

_BT_
