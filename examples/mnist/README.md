# Basic MNIST Example

```bash
pip install -r requirements.txt
python main.py
# CUDA_VISIBLE_DEVICES=2 python main.py  # to specify GPU id to ex. 2
```

### ScheduleFree

```
Test set: Average loss: 0.0367, Accuracy: 9873/10000 (98.73%)
Test set: Average loss: 0.0288, Accuracy: 9896/10000 (98.96%)
Test set: Average loss: 0.0273, Accuracy: 9907/10000 (99.07%)
Test set: Average loss: 0.0248, Accuracy: 9926/10000 (99.26%)
Test set: Average loss: 0.0257, Accuracy: 9930/10000 (99.30%)
Test set: Average loss: 0.0268, Accuracy: 9929/10000 (99.29%)
Test set: Average loss: 0.0268, Accuracy: 9921/10000 (99.21%)
Test set: Average loss: 0.0275, Accuracy: 9929/10000 (99.29%)
Test set: Average loss: 0.0279, Accuracy: 9931/10000 (99.31%)
Test set: Average loss: 0.0278, Accuracy: 9933/10000 (99.33%)
Test set: Average loss: 0.0274, Accuracy: 9935/10000 (99.35%)
Test set: Average loss: 0.0278, Accuracy: 9936/10000 (99.36%)
Test set: Average loss: 0.0289, Accuracy: 9938/10000 (99.38%)
Test set: Average loss: 0.0304, Accuracy: 9935/10000 (99.35%)
```

### Default PyTorch Implementation

```
Test set: Average loss: 0.0476, Accuracy: 9836/10000 (98.36%)
Test set: Average loss: 0.0337, Accuracy: 9889/10000 (98.89%)
Test set: Average loss: 0.0338, Accuracy: 9893/10000 (98.93%)
Test set: Average loss: 0.0310, Accuracy: 9891/10000 (98.91%)
Test set: Average loss: 0.0285, Accuracy: 9908/10000 (99.08%)
Test set: Average loss: 0.0284, Accuracy: 9909/10000 (99.09%)
Test set: Average loss: 0.0270, Accuracy: 9915/10000 (99.15%)
Test set: Average loss: 0.0273, Accuracy: 9914/10000 (99.14%)
Test set: Average loss: 0.0268, Accuracy: 9921/10000 (99.21%)
Test set: Average loss: 0.0258, Accuracy: 9920/10000 (99.20%)
Test set: Average loss: 0.0264, Accuracy: 9919/10000 (99.19%)
Test set: Average loss: 0.0259, Accuracy: 9919/10000 (99.19%)
Test set: Average loss: 0.0260, Accuracy: 9920/10000 (99.20%)
Test set: Average loss: 0.0262, Accuracy: 9922/10000 (99.22%)
```