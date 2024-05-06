# Running experiments
Training script is run as follows:

```python train.py --configs <1 or more paths to .yaml files> --values wandb=True output_path=<path> model_type=VICReg vicreg.base_lr=42```

Important configuration options:
- `model_type`: defines which model to use, options are `VICReg`, `RSSM`, `SimCLR`.
- `dataset_noise`: controls changing noise level
- `dataset_static_noise`: controls fixed noise level
- `dataset_structured_noise`: controls whether the noise is structured.
- `output_path`: specifies the path to save models.
- `wandb`: enables weights and biases logging.
- `dataset_type`: chooses between one dot and three dot datasets. For one dot set `Single`, for three set `Multiple`.

To access particular model's options set options through the respective subconfig, e.g. `vicreg.base_lr`.

# Reproducing results
All configs are saved in `reproduce_configs` folder. To run a config from that folder, you can run

```python train.py --configs reproduce_configs/sweep_fixed_uniform.(1.25).vicreg.best.yaml```

This will run the best VICReg configuration for fixed uniform noise with coefficient 1.25.

# Personal Exploration Notes

## 2024/05/05 log (VICRegPredMultistep class in `vicreg.py`):

The forward pass is fed states and actions. States are pictures, actions are coordinate movements. The idea, as explained in the paper, is to track the movement of a dot across time steps in simple images in latent space.

I am using the best configuration provided in the README above. I made a few edits - for example, they play around with adding noise of some kind, but for the sake of these notes I turned those off. Otherwise, this is tracking the series of events with those configurations.

### States

States are each 28x28 grayscale images. They come in sequences of 17 images each, and will refer to these as "timesteps" in the future. Each has the dot, and you can see the dot moves in a flowy pattern over time, as seen below:

![Figure_1](https://github.com/CCranney/JEPA_SSL_NeurIPS_2022/assets/11773171/6232cb03-ce31-4c71-bd7c-8e5cc84b3aee)

**NOTE: the data is manipulated right after the epoch for loop in `train.py` before being fed into the forward pass.**

1. Straight from the data generator before the forward pass, these states have the shape `torch.Size([32, 17, 1, 28, 28])`. These represent the batch size, timesteps, an unknown variable, height, and width of the images. The unknown variable is probably number of colors, which is only 1 with grayscale.
2. The dimensions are shuffled around to shape `torch.Size([17, 32, 1, 28, 28])` before entering the forward pass. Note that the timesteps and batch sizes were swapped.
3. (in the forward pass) All the images are basically concatenated into one long tensor of size `torch.Size([544, 1, 28, 28])`. Note the batch size x timestep dimension at the front.
4. The states are encoded (using the model represented by the `backbone` class variable). This is variable, but the developer used a MeNet5 architecture (outlined in the `models.py` file). The output is of shape `torch.Size([544, 512])`. To be investigated in the future.
5. The timestep and batch sizes are again differentiated, so the shape is `torch.Size([17, 32, 512])`.
6. This is where I stopped for the day.

### Actions

Actions are 2-variable inputs. I believe they correspond to the changes of the X and Y positions of the dot. Not sure if it's counted from the top left or the bottom left. Because there are 17 states, there are 16 actions connecting state to state.

Here's a printout of the actions **corresponding to the above picture:**

```
	[-0.9991230964660645, 0.4641880691051483]
	[0.06517060846090317, 0.004006990231573582]
	[0.13474783301353455, -1.1007559299468994]
	[-1.0126862525939941, -0.2678211033344269]
	[-1.574415922164917, -0.848552942276001]
	[1.6843935251235962, 0.5218906402587891]
	[-0.6482175588607788, 0.08043285459280014]
	[0.754325270652771, 0.2969764769077301]
	[-1.4604908227920532, 0.31641003489494324]
	[1.0131354331970215, 0.03856400400400162]
	[-0.4284696578979492, 0.03630372881889343]
	[-0.4723133146762848, 1.3543951511383057]
	[-0.37998002767562866, -0.30847692489624023]
	[-0.08391612768173218, 0.9859762191772461]
	[-2.11621356010437, 1.0932966470718384]
	[-0.13971057534217834, 0.015490273013710976]
```

1. Straight from the generator before the forward pass, actions are of shape `torch.Size([32, 16, 1, 2])`. These represent the batch size, the number of actions or timesteps - 1, an unknown variable, and the number of directions the dot can move. The unknown variable may be to allow for the possibility of multiple dots?
2. Like the states, the shape of the action array is shuffled around to be of shape `torch.Size([16, 32, 1, 2])`. Note that timesteps-1, batch_size have been swapped.
3. The unknown variable is unsqueezed out, producing the shape of `torch.Size([16, 32, 2])`.
4. Actions are not passed through the `backbone` model, as expected, but thought I'd point out the difference in treatment between states and actions (and their corresponding shape changes) up to this point.
5. I did not get to any action manipulations in the forward pass today.

