## For reorganising the structure of modules

Each encoder, instead of being its own thing, will now be under EncoderModule, which is a module that manages the encoders as a whole.
In other words, the agent calls EncoderModule's forward with some kwargs, and EncoderModule should do that on its own

Same thing with WM and Decoders and policy-head and Loss stuffs. Although WM is relatively simple since we arent planning to have many(?), just one

And Loss / optim stuff comes later since its probably not so important
