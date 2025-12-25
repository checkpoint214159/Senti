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

# 17/12/2025 Atomic system

instead of only enumerating in terms of 1 timestep at a time, its much better to abstract to arbitrary number of timesteps under one 'atomic timestep', which is the largest timestep size within which our agent will operate on as a unit timestep dimension. much more flexible and a pretty darn good abstraction if i do say so myself :^D

# 18/12/2025 including Z in vfe state calculation

from the earlier 14/12 log where i describe what the wm is predicting, it also makes sense to treat the return of the model as 'the prediction for z for the next timestep', and since z and s are basically semantically relevant to each other (z is converted into s in the qkv modules of the transformer) including it actually makes our VFE calculations work!

However, I appear to have hastily created this extra fucntionality, without caring about scalability or encapsulation! oh well, future me's problem. should be a one-day fix only where i just ponder the responsibilities of the various componenets yet again as i sip on a monster

# 18/12/2025 LOGGING??

this has been a long time coming, but i should do a lot more and better logging. because I havent even started on planning / policy semantics yet, and its already so complex, with many things to possibly go wrong, and I havent written tests yet (but do I need to? yes yes of course what the hell am i asking that for)

i should make a seperate branch for that since it will be incrementally implemented as i dont assume it will be immediately apparent to me how to best do this


# 19/12/2025 Chapter 2: Planning

I have taken approximately 10 months (2 months equivalent of intensive work) to set up this whole repo and its structure, having refined my understanding of program design and also the architecture with which I am researching with. This has largely been spent on the 'zeroth + first part' of the project, that is creating an agent that infers. Not very well so far (obviously since it hasnt been trained), but inference and all the structure around it exists.

Now, it is time for part 2; creating the planning module, which for all I know, is one-of-a-kind in RL-like environment research. Let me just elaborate:

Policy and action prediction: While the semantic of 'policy' is vague and left up to implementation, it roughly translates to 'something that guides the selection of action of the agent'. I will refine it in our case to be "A set of parameters, that when applied to something (probably our beliefs over state, z or h), predicts a belief over what actions the agents will take". Note how I dont say "what actions are best", since as per the philosophy of this project/research, the idea of 'best'/'ground truth' is irrelevant, and what unfolds based on our evolutionary parameters is the optimally selected for (and this selection is stochastic and human-removed, so its not really a truth as it is just reality).

To contextualize it further, I will define it to be a function pi over z, that is a = pi(z) (wont pass in h, clear up the exact semantics between h and z a little more next time).

Inference:
The agent recieves observations, converts it to a belief, updates its own beliefs (and does some parametric learning too, in the autoencoders + worldmodels). So far so good, except for the fact that I havent considered whether action is updated or learned here (it shouldnt be, but i should make 100% sure)

Planning
After inference, the agent samples a number of parametric policies that define some kind of or part of a 'policy head', which is the aforementioned pi. (after a = pi(z), we call new_x, new_h = transition(a, z, h), and then new_x is mapped to new_z then new_a. (TODO: define the semantics better next time, about which is predicted by what and when). )
The core difference is that this is our rollouts, and after we reach some future stopping timestep, we verify against our grounding_wm, which acts as our prior here. This comparison with prior may take the form of risk or pragmatic value (go read EFE mathematical decomposition for specifics).

Then, for the semantics arounding ambiguity / epistemic value, which basically describes a term contributing to an agent's exploratory instincts, to minimize the unknown (reducing ambiguity) == maximize what it knows (increasing epistemic knowledge).


# 19/12/2025 Semantics of transition vs grounding wm

I think its important to clarify the semantic differences between the two. First, consider the actions that these modules consume, during inference and planning

Inference:
Transition model does not predict any action when we create a new state, as the previous action taken post-planning is the action we took

And when calculating VFE during inference steps, action does nto change, as what action we took, is what we took, no matter what we believe (? maybe not? what happens if we allow the agent to adjust its belief over tis actions??)

GWM: Posesses the learned, 'default' policy head, which learns to map the z it predicts to a predicted action, then backprops against the actual actions taken to learn the policy head. In this sense, it encapsulates some essense of the agent's phenotype, that is a part of it that learns to act in the way 'that it already has'.


Planning:
Transition model: Sample a variety(?) of policies from some belief over policies (is this the belief derived from default policy?), compute EFE, and backprop to update belief over policy, once stabilised / terminal step reached, use the updated policy to predict the next action and act

GWM: To create priors over future states, what is 'prior' is our default belief, including our default policy belief. This remains static and detached from the policy belief used by the transition model during planning, even after performing EFE computation and updating the policy belief used by the transition model, this remains static, only updating itself during inference

# 20/12/2025 Action & Policy-head design

I think I will reframe the "action head" of a model to be a function f(z, gene, u), where the 'gene' refers to some kind of 'preferences prior' (lets call it an inductive bias) that is evolutionarily learned, and u is a parameter from some not-so-large dimensional space that has the semantic of a 'policy'. In GWM, this is the default policy that is learned via inference. In the transition model, this will be sampled during planning.

# 22/12/2025 Policy update cycle

Here is something I figured after yapping and bouncing off Gemini for a bit:

1. Start with a policy that is initialized based off the Gene
2. During inference, do not update policy
3. During grounding wm, use the 'current policy', which at t=1 is the policy inited off the Gene. This will be updated to reflect the post-inference priors
4. During planning, sample from this policy, and do rollouts. Calculate EFE and whatnot, backprop to update this policy.
5. The updated policy and the pre-updated policy are combined to get the 'current policy', via 'temporal smoothing':
$u_{prior} = (1 - \lambda) \cdot u_{GWM}(z_t) + \lambda \cdot u^*_{t-1}$
6. Sample the next action ONLY from the updated policy

word

# 22/12/2025 Big action-wise TODO list

TODOs for action related semantics

1. Make some action autoencoder, so we can map to and from action space -> latent action space
2. Decide on 'containerising' the wm class and action head class inside a new one
3. 'action' belief updating too

yeah wait thats about it

# 23/12/2025 BIG SEMANTIC CHANGE!!!!

shits cooked gang, we gonna have to ditch our old h,z,a interpretation and use Dreamer's. I originally used the openai vpt minecraft model architecture cuz of the original scope, but for active inference frankly dreamer makes 10x more sense.

h: Deterministic state. Predicts z_t, is predicted via h_t = f(h_t-1, z_t-1, a_t-1)
z: Stochastic 'belief' state. h predicts z via p(z|h) (or also p(z|h, o))
a: Action.

We will now use RSSM instead of the stupid wm from openai. thanks openai


# 24/12/2025 Conv1d

Merry Christmas Eve! I will use nn.Conv1d to squash down the timestep dimension from [B, T, E] tp [B, E], which makes it easier to operate on our GRU cells, and also makes it cleaner if we will operate at variable timesteps (so each level of timesteps, will have its own conv that aggregates whatever T is to 1, then we unsqueeze that dim)


# 25/12/2025 Preferences and genome

To reiterate, genome is something that is optimized by evolution, and is treated as static and not learned within the agent's lifespan. 

Anyways during planning, since we do not have observations nor rewards to help evaluate the instrumental value of our rollouts / imagined futures, we will use a "preference head", conditioned on h and genome. This spits out a z that we take reference from, which describes a "desired z". This is what we will use to derive instrumental value, which is what we need for EFE calculation


