## Update notes
for me to keep track of things and ideas that I am halfway implementing. maybe lets all write blog posts about this :D

### For reorganising the structure of modules

Each encoder, instead of being its own thing, will now be under EncoderModule, which is a module that manages the encoders as a whole.
In other words, the agent calls EncoderModule's forward with some kwargs, and EncoderModule should do that on its own

Same thing with WM and Decoders and policy-head and Loss stuffs. Although WM is relatively simple since we arent planning to have many(?), just one

And Loss / optim stuff comes later since its probably not so important

#### Inspiration for the above

From my previous experience with MMEngine, which is honestly a mega-goated pipeline structure that I learned so much from (modularity, registry, config things) and I want to build on / improve for my specific use case (imo its already really good but just a little unwieldy in its full heft)

### 30/11/2025 Dataclasses
tired of dealing with the transition model's nested tuple list structure, very obnoxious to have to pack and unpack it for my use. I will forcibly make some of the early interfaces use a more descript dataclass, and unpack it for the inner workings stuff that I will not touch as much

also good to create dataclasses for state, into various things like Discrete and Normal distributions. for re-use of functionality of parent classes. man OOP *when done right* is truly magical.

### 5/12/2025 Not learning variance for now?
Havent began feeding variance to the models, since I'm not sure if I want that or how should I approach the problem of learning the variance tensor for now

### 5/12/2025 Decentralising loss?
This isn't some crypto web 3.0 decentralised shit;
Since loss in several of our cases is only calculated between one timestep and another, there is no global 'gather and backprop' thing in our system for these cases (still exist for our system just not for these cases), can we attach the functions that define, and functions that calculate loss, onto some dataclass? Then, given the dataclass obj i will merely pass it another, and it will calculate it for me

this should be more scalable and neater than the current approach of promising all the calculation and then doing it all at once in a step (which once again is fine, just maybe unneeded for the above case)

idk what im yapping about bruh


### 5/12/2025 Boundaries between models and dataclasses
I think its best if I leave models independent of my dataclasses, and simply create a child of them to adapt it to my use case. Yk keep that abstraction layer there
 
### 6/12/2025 Device casting
Oh god I dont want to write so many .device all over the place for everything that is an nn.Module. TODO find a better way to do this.

### 13/12/2025 extend ### 5/12/2025 Not learning variance for now?
i think what I will do is, leave a clear NOTE signature for all the exact places that I explicitly extract a mean and compute using it, so I can easily search-all and find the relevant lines, if (or basically when) i need to make changes to this workflow

I think the note will be # NOTE: LEARN VARIANCE BOOKMARK HERE

also its been a short while where have i been? learning web dev with the help of gemini man this holiday is more productive than my school life lolol. anyways while doing that I was also touching up on the worldmodel stuff and wrapping more things around dataclasses functionality.

# 14/12/2025 need to write down before I go insane

I think I need to write down all of this before I go insane.
I HAVE BEEN THINKING OF THE WORLDMODEL WRONG.
In fact, its been a very typical decoder-only transformer model. but let me write all of this down for much easier future reference. This will be ultra-long, so strap in.

### The goal of the 'transition model' / 'world model'.
The goal of the transition/world model has always been to predict "what comes next". Specifically in our case, we have some completely time-dependent latent observation z, and some 'generalized hidden state' h that encodes information across many timesteps. h is derived from z, but also not really, but kind of, just for now take them as very very related.

lets say I am at timestep t, and I want to predict the hidden state at t+1
given my current hidden state h_t and incoming observation z_(t+1). I then encode z_(t+1) into h_t+1, and voila.
...Wait. so where does h_t even come into play? this is where we must discuss the specifics of the model. specifically, this decode-only transformer model with masked attention + recurrent functionalities

### Talking about architecture without talking about architecture.

For all intents and purposes, treat the transition/world model as a function over two things: state, and x. state is equivalent to h, and x is equivalent to z. it also returns those two things. Lets talk about what actually goes on behind the scenes:

state has length n, like a list of length n, where each element denotes a distinct 'state'.
model:
    - recurrent_layer 1
    - recurrent_layer 2
    - recurrent_layer 3
    - recurrent_layer ...
    - recurrent_layer n
lastly, consider our model operating on a timestep scale of 1. ignoring other dimensions like batch and embedding dimension. Usually our tensors look like this shape (B, T, D), we are pretty much only focusing on the middle one.

say I have some state s0, and x x1. We denote with an underscore _dn, that this semantic exists at depth n. 
What the model does is, at the first recurrent_layer, d=1:
1. encode x1_d0 -> s1_d1.
2. encode x1_d0 -> query
3. using s0_d1 and s1_d1 as kv, and the above query, do masked-causal self-attention.
4. predicts x1_d1
note from the above, I am not predicting some s1_d1 BASED on s0_d1. However, the next part is key:

at the next recurrent layer, d=2:
1. encode x1_d1 -> s1_d2.
2. encode x1_d1 -> query
3. using s0_d2 and s1_d2 as kv, and the above query, do masked-causal self-attention.
4. predicts x1_d2
aha! it took the previous output, fed it to the next layer, which influenced the s1_d2 creation. in other words and in very handwavy notation:
s0_d1, x1_d0 ~> s1_d2, i.e s0_d1 does NOT contribute to s1_d1, like s0_d1, x1_d0 ~> s1_d1.

And thus we reach the true description of what state is, and what it entails. Each depth to state denotes some more meaningful, previous-timestep dependent representation of previous states.

Also note, that x doesnt have a list of values of various depth, the _dn semantic jsut refers to some meaning carried with it, not the fact that x actually has a series of depth values.

### What happens after?
so what happens to x1_d2 or x1_dn, however many layers there are? well, we feed it to a prediction head, which maps it to the semantic of 'what is next', basically x2. and thus we have described a decoder only architecture!

so here is a final reminder: if s0 and x1 are fed into the model, you get (s0 and) s1, and x2!
