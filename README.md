# Valohai TensorFlow Examples

This repository serves as an example for the [Valohai MLOps platform][vh]. It implements handwritten digit detection using
TensorFlow, based on [TensorFlow's example][ex].

The project describes a machine learning pipeline with five unique steps:
- Preprocess data
- Train model
- Batch inference
- Compare predictions
- Online inference deployment

If you are just starting out, it is recommended to follow [the learning path][lp] in the Valohai documentation. This learning path recreates the Train model step of this example.

[ex]: https://www.tensorflow.org/tutorials/quickstart/beginner
[vh]: https://valohai.com/
[lp]: https://docs.valohai.com/tutorials/learning-paths/fundamentals/valohai-utils/

## Running on Valohai

Login to the [Valohai app][app] and create a new project.

In the project's settings, configure this repository as the project's repository.

Now you are ready to run executions, tasks and pipelines. After you've trained a model, you can also deploy it for online inference!

[app]: https://app.valohai.com

## Running Locally

You can run all the steps of the pipeline locally. This requires Python 3.9 and specific packages, which you can install with:

```python
pip install -r requirements.txt
```

The steps require different inputs to run, so you need to run them in order.

Preprocess data has all the required inputs defined as defaults and can be run with:
```python
python preprocess_dataset.py
```

Train model requires the preprocessed dataset, but that is also defined as a default, so you can run:
```python
python train_model.py
```

Batch inference requires both a model and some new data. The new data has default values, but the model needs to be provided, for example from an earlier train model run:
```python
python batch_inference.py --model .valohai/outputs/{local_run_id}/train-model/model-{suffix}.h5
```

Compare predictions requires two or more batch inference results and optionally the corresponding models. We can run it for example like this:
```python
python compare_predictions.py --predictions .valohai/outputs/{local_run_id}/batch-inference/predictions-{suffix}.json .valohai/outputs/{local_run_id}/batch-inference/predictions-{suffix}.json
```
