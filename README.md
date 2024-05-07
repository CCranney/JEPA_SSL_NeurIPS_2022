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

## 2024/05/06 log (VICRegPredMultistep class in `vicreg.py`):

Continuing my evaluation of the forward pass. I reached this line in the function last time, and will be continuing from there:

`all_states_enc = all_states_enc.view(*states.shape[:2], -1)`

### Encoder and Expander

This is diverging a bit from what I would expect. I'm going to give a high-level overview of what is happening and go from there.

If you'll recall from the diagrams of the original JEPA paper, there are generally three neural net bottlenecks of the JEPA architecture. The first of those is at least one encoder/perceiver, which encodes some kind of input. X and Y are encoded separately, and theoretically can be from different mediums and therefore encoded differently, but all instances of JEPA I am familiar with use the same encoder for both X and Y. In this code the backbone is the encoder, or perceiver, as described in the original JEPA paper. 

In this code, the encoder is applied to all 17 states right at the beginning. 

A projector object is also used. It is defunct in this current run of code, but I believe it is meant to be the expander of Figure 14 "Training a JEPA with VICReg" in the original JEPA paper. I believe it's purpose is to expand the number of features represented by the encoded representation. This would in turn enhance the VICReg regularization method, which tries to distinguish samples across a batch to avoid model collapse. It makes the encoded data bigger and more detailed, basically. 

The projector/expander in this instance is just an nn.Identity() module, which simply passes the input forward as output (it's an empty placeholder). It allows, however, for a simple MLP to be implemented instead.

So, continuing from the above **states** breakdown:

1. The current encoded state object shape is `torch.Size([17, 32, 512])`.
2. Some allowance is made for "burnin" variables to be made for both actions and states. Burn-in is a principle of "warming up" an RNN before training is initiated. In this case, these principles are ignored, though such features have been implemented if desired.
3. The encoded state is passed through the projector. Again, in the present code, this does nothing.

### Predictor

The first state is the only true X of the dataset. The program then consists of an GRU RNN (the predictor) that tries to predict each state sequentially, only using the next action and the previous estimated state to predict each new state. The actions are therefore considered the input. A loss is computed between each prediction and it's corresponding actual encoded state. In short, the first state is X, and following states are Y at their own timestep. **This means that the other states are never actually fed into the model directly.** In many ways, the program is trying to predict the end from the beginning in a hierarchical fashion. See the `RNNPredictor` class in `model.py` for more details.

This is all explained by Figure 1 of the paper, the left-hand side. I just didn't realize I was looking at an RNN when I saw it the first time. The output at each timestep is used for direct comparison with the actual state at that timestep for the loss function.

1. The first encoded state (`current_enc`) and all of the actions are fed into the predictor module with the `self.predictor.predict_sequence` line. Ignore the h0, that's an ignored element of the burn-in bit explained earlier.
2. NOTE: I just realized, but the first state fed into the RNN is not projected at all. Only subsequent states were.
3. The output is of shape `torch.Size([16, 32, 512])`. These are the predictions made by the model, one for each sequential action.
4. Notably, the predictions are also fed to the projector before comparison with the actual states (NOT during the RNN itself). Again, this does nothing in this instance, but that would apply if the projector was an actual MLP.
5. Sometimes the projection is ignored, depending on user input (depends on value of `self.args.repr_loss_after_projector`).


### Losses

Something I was kind of scratching my head over - This was supposd to preform VICReg loss functions, but wasn't seeing a reference to such loss in the places I would expect. It all happens at the end of the forward function.

#### Invariance / Normal Loss

1. The MSE loss between the predicted projected states (`pred_proj`) and the actual projected states (`states_proj`) is calculated. Each of them are added to a total loss saved in the variable `repr_loss` (representative loss, I assume).
2. There is also the option of using the current state and the next state to predict the action (depending on the value of `self.args.action_coeff`). This prediction is a pretty simple linear MLP, no activation function, predicting the 2 floats of the activation from the concatenation of those two states.
3. If chosen, there is also an action loss saved to `action_loss`.
4. Both `repr_loss` and `action_loss` are normalized by dividing by 16 (timesteps-1).

#### Reconstruction Loss

It looks like you have the option of decoding the encoded states (not the predicted or projected states) and calculating a loss respective to that decoding process.
Pretty simple:

1. Flatten the encoded states
2. Run through a decoder (`VAEDecoder` in this case).
3. Calculate the loss between the decoded states and the original states.
4. There's also a visualization option available.

#### Variance and Covariance Loss

Here we go. 

1. All encoded and projected states (NOT predicted states!) are zipped up and looped over with `for enc, proj in enc_projs`.
2. Depending on user input, this can skip the first encoded/projected state (see `self.args.skip_first_step_vc`).
3. There are various if/else statements here, but the variance and covariance are calculated using `get_cov_std_loss(proj)`.
4. Some settings let you skip running `get_cov_std_loss(enc)`, which makes sense in this case. No projection occurred, so this shouldn't be run for both.
5. The covariance and I assume variance (`std` probably being standard deviation) loss is added to a growing total for each.

#### Final Loss Summary

1. All the losses are added together.
2. Notably, the user can set various coefficients for each of these losses to selectively reduce their impact.
3. All losses are returned in a `LossInfo` class alongside the total loss. The non-total losses are put in a `diagnostics.VICRegDiagnostics` class, which I assume is used for tracking them for W&B metrics.

And that's the workflow in a nutshell.

