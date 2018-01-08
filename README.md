# SimulatedAnnealing
Pytorch Optimizer for Simulated Annealing

## Usage
You need to define a sampler, eg:

```python
sampler = UniformSampler(minval=-0.5, maxval=0.5, cuda=args.cuda)
# or
sampler = GaussianSampler(mu=0, sigma=1, cuda=args.cuda)
```

The sampler is used for the annealing schedule for Simulated Annealing.
The optimizer is a standard pytorch optimizer, however you need to pass a closure into the `step` call:
```python
optimizer = SimulatedAnnealing(model.parameters(), sampler=sampler)
def closure():
    output = model(data)
    loss = F.nll_loss(output, target)
    return loss

optimizer.step(closure)
```
