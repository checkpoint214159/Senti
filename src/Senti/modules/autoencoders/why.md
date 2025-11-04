# Why autoencoders?

Simply because it is a "convenience". The "Auto" part comes from the ancient way of training unsupervised, by encoding, then decoding to reconstruct, then comparing with the original (before encoding)

Because this project will attempt to "find interesting behaviour within the agent", one area we can monitor is the decoded "reconstruction" of latent state. E.g during planning, lets say with the agent using its transition model to predict the next world state and the next etc etc, we can decode the observational side of it and re-project it back to our environment's observation space, thus literally seeing "the plan" of the agent. This visualization would be greatly helpful to us!

## Don't initialize either component if you dont want it

We will just enable the choice of initing the decoder or encoder, since the alternative is just a "one-way-mapping" from latent to observation (or observation to latent) which is a subset of an "Autoencoder's" functionality.


### what is resources doing there

uhh idk how it suddenly appeared. probably a bad accidental drag. will see if anything is broken then put it back where it belongs. kinda like a lost child tbh
