# PyTorch Trainer

To train neural network models using PyTorch, we construct a PyTorch Trainer - inspired by [HuggingFace's Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) - 
to orchestrate the complex pipeline of PyTorch models creation, training and evaluation. This README shall motivate the 
choise of this implementation.  

## Why We Need a Trainer Class?

CY-Bench models use a unified interface where all models follow the same pattern:

```python
model.fit(dataset, **fit_params)
predictions, info = model.predict(dataset, **predict_params)
```

This works seamlessly for sklearn, XGBoost, and other ML libraries. However, PyTorch is fundamentally different—it's a **deep learning framework**, not just a model library.

Unlike `sklearn_model.fit()`, PyTorch requires orchestrating multiple components:
- **Training loops** over epochs and batches
- **DataLoaders** for streaming data
- **Optimizers** (Adam, SGD) for gradient updates
- **LR Schedulers** for learning rate adjustment
- **Loss functions** for computing gradients
- **Device management** (CPU/GPU)
- **Checkpointing** model and optimizer states

The **Trainer class** encapsulates this complexity while exposing the same simple `.fit()` and `.predict()` interface as other models. 

```python
trainer = TorchTrainer(
    model=,
    optimizer=,
    scheduler=,
    loss_fn=
)

trainer.fit(dataset)
predictions, info = trainer.predict(dataset)
```
⚠️ Possible Confusion: The Trainer class inherites from BaseModel (models/model.py) but itself is not a PyTorch model. The property `trainer.model` is a PyTorch model.
